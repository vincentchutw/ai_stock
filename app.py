"""
【Code Gym】AI 股票趨勢分析系統
使用 Streamlit + FMP API + OpenAI o4-mini 進行專業技術分析
"""

import json
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta

# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 主標題 ────────────────────────────────────────────────────────────────────
st.title("📈 AI 股票趨勢分析系統")
st.divider()  # rainbow divider in newer Streamlit; fallback gracefully


# ══════════════════════════════════════════════════════════════════════════════
# 核心函數
# ══════════════════════════════════════════════════════════════════════════════

def get_stock_data(symbol: str, fmp_api_key: str) -> pd.DataFrame | None:
    """從 FMP API 獲取股票完整歷史數據（含分頁處理）"""
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={symbol.upper()}&apikey={fmp_api_key}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # FMP 回傳格式：直接為 list 或 {"historical": [...]}
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "historical" in data:
            records = data["historical"]
        else:
            st.error(f"❌ 找不到股票 **{symbol}** 的數據，請確認股票代號是否正確（例：AAPL、MSFT、GOOGL）。")
            return None

        if not records:
            st.error(f"❌ 股票 **{symbol}** 沒有可用的歷史數據，請確認代號是否正確。")
            return None

        df = pd.DataFrame(records)
        # 統一欄位名稱（FMP 可能返回 camelCase 或 snake_case）
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except requests.exceptions.ConnectionError:
        st.error("❌ 無法連接到 FMP API，請檢查網路連線。")
    except requests.exceptions.Timeout:
        st.error("❌ FMP API 請求超時，請稍後再試。")
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            st.error("❌ FMP API Key 無效，請確認金鑰是否正確。")
        else:
            st.error(f"❌ FMP API 錯誤：{e}")
    except Exception as e:
        st.error(f"❌ 獲取數據時發生錯誤：{e}")
    return None


def filter_by_date_range(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """根據用戶選擇的日期範圍過濾數據"""
    mask = (df["date"] >= pd.Timestamp(start_date)) & (
        df["date"] <= pd.Timestamp(end_date)
    )
    filtered = df[mask].reset_index(drop=True)
    return filtered


def get_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """計算 MA5、MA10、MA20、MA60 移動平均線"""
    df = df.copy()
    for window in [5, 10, 20, 60]:
        df[f"ma{window}"] = df["close"].rolling(window=window, min_periods=1).mean()
    return df


def generate_ai_insights(
    symbol: str, stock_data: pd.DataFrame, openai_api_key: str
) -> str | None:
    """使用 OpenAI o4-mini 進行技術分析"""
    # 準備傳給 AI 的 JSON 數據（取最近 90 筆，避免 token 超限）
    cols = ["date", "open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "ma60"]
    available_cols = [c for c in cols if c in stock_data.columns]
    data_slice = stock_data[available_cols].tail(90).copy()
    data_slice["date"] = data_slice["date"].dt.strftime("%Y-%m-%d")
    data_json = data_slice.to_json(orient="records", force_ascii=False)

    first_date = stock_data["date"].iloc[0].strftime("%Y-%m-%d")
    last_date = stock_data["date"].iloc[-1].strftime("%Y-%m-%d")
    start_price = float(stock_data["close"].iloc[0])
    end_price = float(stock_data["close"].iloc[-1])
    price_change = ((end_price - start_price) / start_price) * 100

    system_msg = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

    user_msg = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% (從 ${start_price:.2f} 變化到 ${end_price:.2f})

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量和移動平均線：
{data_json}

### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 4. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 5. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）
- 關鍵價位觀察點
- 技術面風險因子

### 綜合評估要求
#### 輸出格式要求
- 條理清晰，分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測，強調分析的局限性
- 在適當位置使用表格或重點標記

分析目標：{symbol}"""

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "auth" in err.lower() or "api key" in err.lower() or "401" in err:
            st.error("❌ OpenAI API Key 無效，請確認金鑰是否正確。")
        elif "quota" in err.lower() or "429" in err:
            st.error("❌ OpenAI API 額度不足或請求過於頻繁，請稍後再試。")
        else:
            st.error(f"❌ AI 分析發生錯誤：{e}")
        return None


def draw_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """使用 Plotly Graph Objects 繪製 K 線圖 + 移動平均線 + 成交量"""
    first_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(
            f"{symbol.upper()} 股價 K 線圖（{first_date} ~ {last_date}）",
            "成交量",
        ),
    )

    # K 線圖
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K線",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # 移動平均線
    ma_config = {
        "ma5":  {"color": "#FFD700", "dash": "solid",  "width": 1.5},
        "ma10": {"color": "#FF6B6B", "dash": "solid",  "width": 1.5},
        "ma20": {"color": "#74C0FC", "dash": "solid",  "width": 1.8},
        "ma60": {"color": "#B197FC", "dash": "solid",  "width": 2},
    }
    labels = {"ma5": "MA5", "ma10": "MA10", "ma20": "MA20", "ma60": "MA60"}
    for col_name, cfg in ma_config.items():
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[col_name],
                    name=labels[col_name],
                    line=dict(color=cfg["color"], dash=cfg["dash"], width=cfg["width"]),
                    opacity=0.9,
                ),
                row=1, col=1,
            )

    # 成交量長條
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="成交量",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 側邊欄輸入區
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 分析設定", divider="rainbow")

    symbol_input = st.text_input(
        "📌 股票代碼",
        value="AAPL",
        help="輸入美股股票代號，例如：AAPL、MSFT、GOOGL、TSLA",
        placeholder="e.g. AAPL",
    ).strip().upper()

    fmp_key = st.text_input(
        "🔑 FMP API Key",
        type="password",
        help="前往 https://financialmodelingprep.com 免費申請 API Key",
        placeholder="貼上您的 FMP API Key",
    ).strip()

    openai_key = st.text_input(
        "🤖 OpenAI API Key",
        type="password",
        help="前往 https://platform.openai.com 取得 API Key",
        placeholder="貼上您的 OpenAI API Key",
    ).strip()

    st.markdown("---")
    default_end = datetime.today()
    default_start = default_end - timedelta(days=90)

    start_date = st.date_input("📅 起始日期", value=default_start)
    end_date = st.date_input("📅 結束日期", value=default_end)

    st.markdown("---")
    run_btn = st.button("🚀 分析", use_container_width=True, type="primary")

    # 免責聲明
    st.markdown("---")
    st.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")


# ══════════════════════════════════════════════════════════════════════════════
# 主程式邏輯
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    # ── 輸入驗證 ─────────────────────────────────────────────────────────────
    has_error = False

    if not symbol_input:
        st.error("❌ 請輸入股票代碼（例：AAPL、MSFT、GOOGL）。")
        has_error = True

    if not fmp_key:
        st.error("❌ 請輸入 FMP API Key。前往 https://financialmodelingprep.com 免費申請。")
        has_error = True

    if not openai_key:
        st.error("❌ 請輸入 OpenAI API Key。前往 https://platform.openai.com 取得。")
        has_error = True

    if start_date > end_date:
        st.error("❌ 起始日期不能晚於結束日期，請重新選擇。")
        has_error = True

    if has_error:
        st.stop()

    # ── 獲取數據 ──────────────────────────────────────────────────────────────
    with st.spinner(f"📡 正在從 FMP 獲取 **{symbol_input}** 歷史數據..."):
        raw_df = get_stock_data(symbol_input, fmp_key)

    if raw_df is None:
        st.stop()

    # 過濾日期範圍
    st.info(f"✅ 成功取得 {symbol_input} 完整歷史數據，共 {len(raw_df):,} 筆，正在過濾日期範圍...")
    filtered_df = filter_by_date_range(raw_df, start_date, end_date)

    if filtered_df.empty:
        st.error(
            f"❌ 在 {start_date} ~ {end_date} 期間找不到 {symbol_input} 的數據，"
            "請嘗試調整日期範圍。"
        )
        st.stop()

    if len(filtered_df) < 5:
        st.warning(
            f"⚠️ 選定期間內僅有 {len(filtered_df)} 筆數據，建議選擇更長的日期範圍以提升分析品質。"
        )

    # 計算移動平均線
    stock_df = get_moving_averages(filtered_df)
    st.success(f"✅ 數據處理完成，共 {len(stock_df):,} 個交易日，已計算 MA5 / MA10 / MA20 / MA60。")

    # ── 股價 K 線圖與技術指標 ─────────────────────────────────────────────────
    st.subheader("📊 股價 K 線圖與技術指標")
    fig = draw_candlestick_chart(stock_df, symbol_input)
    st.plotly_chart(fig, use_container_width=True)

    # ── 基本統計資訊 ──────────────────────────────────────────────────────────
    st.subheader("📋 基本統計資訊")
    start_price = float(stock_df["close"].iloc[0])
    end_price = float(stock_df["close"].iloc[-1])
    price_diff = end_price - start_price
    price_pct = (price_diff / start_price) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"起始價格（{stock_df['date'].iloc[0].strftime('%Y-%m-%d')}）",
            value=f"${start_price:.2f}",
        )
    with col2:
        st.metric(
            label=f"結束價格（{stock_df['date'].iloc[-1].strftime('%Y-%m-%d')}）",
            value=f"${end_price:.2f}",
        )
    with col3:
        st.metric(
            label="期間漲跌幅",
            value=f"{price_pct:+.2f}%",
            delta=f"${price_diff:+.2f}",
            delta_color="normal",
        )

    # ── AI 技術分析 ───────────────────────────────────────────────────────────
    st.subheader("🤖 AI 技術分析")
    with st.spinner("🧠 AI 正在分析中，請稍候（約 10~30 秒）..."):
        ai_result = generate_ai_insights(symbol_input, stock_df, openai_key)

    if ai_result:
        st.markdown(ai_result)

    # ── 歷史數據表格 ──────────────────────────────────────────────────────────
    st.subheader("📁 歷史數據表格（最近 10 個交易日）")
    display_cols = ["date", "open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "ma60"]
    available_display = [c for c in display_cols if c in stock_df.columns]
    table_df = stock_df[available_display].tail(10).sort_values("date", ascending=False).copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")

    # 數值格式化
    price_cols = ["open", "high", "low", "close", "ma5", "ma10", "ma20", "ma60"]
    for pc in price_cols:
        if pc in table_df.columns:
            table_df[pc] = table_df[pc].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    if "volume" in table_df.columns:
        table_df["volume"] = table_df["volume"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "-"
        )

    table_df.columns = [
        {"date": "日期", "open": "開盤", "high": "最高", "low": "最低",
         "close": "收盤", "volume": "成交量", "ma5": "MA5",
         "ma10": "MA10", "ma20": "MA20", "ma60": "MA60"}.get(c, c)
        for c in available_display
    ]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

else:
    # 歡迎說明頁
    st.info(
        "👈 請在左側填入**股票代碼**、**FMP API Key**、**OpenAI API Key** 及**日期範圍**，"
        "然後點擊「🚀 分析」按鈕開始分析。\n\n"
        "**股票代碼範例**：AAPL（蘋果）、MSFT（微軟）、GOOGL（谷歌）、TSLA（特斯拉）、NVDA（輝達）"
    )
    st.markdown("""
---
### 📖 技術指標說明

| 指標 | 說明 |
|------|------|
| **K 線圖** | 每日開盤、最高、最低、收盤四個價格的視覺化表示，綠色為上漲，紅色為下跌 |
| **MA5** | 5 日移動平均線，反映短期價格趨勢 |
| **MA10** | 10 日移動平均線，反映短中期價格趨勢 |
| **MA20** | 20 日移動平均線（月線），反映中期主要趨勢 |
| **MA60** | 60 日移動平均線（季線），反映長期趨勢方向 |

> **提示**：所有分析內容純屬教育用途，不構成任何投資建議。
""")

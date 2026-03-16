import os
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Sequential


# ==========================================
# 1. 全局配置与严格可复现性设置 (核心要求 2)
# ==========================================
# 强制锁定随机种子，不暴露在 UI 中
def set_reproducibility(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


set_reproducibility(7)
warnings.filterwarnings('ignore')

st.set_page_config(page_title="JPM Chooser Option Pricing Dashboard", layout="wide", initial_sidebar_state="expanded")

# 核心要求 1：保留深色高对比度并固定 5 种模型名称
COLOR_MAP = {
    'Actual_CME_Price': '#FFFFFF',  # 纯白 (真实价格)
    'Paper_Original_BSM': '#FFD700',  # 亮黄 (BSM基础)
    'Hybrid': '#39FF14',  # 荧光绿
    'E2E': '#FFA500',  # 亮橙
    'LSTM': '#FF00FF',  # 品红/荧光粉
    'GRU': '#00FFFF'  # 青色/荧光蓝 (核心残差模型)
}


# ==========================================
# 2. 特征工程统一处理模块
# ==========================================
def prepare_features(df):
    """统一的特征衍生函数，保证训练集、测试集、压力测试集调用一致，避免 KeyError"""
    df = df.copy()

    # 基础金融特征计算
    df['Price_Return'] = df['JPMorgan_Stock_Price'].pct_change().fillna(0)
    df['Moneyness'] = df['JPMorgan_Stock_Price'] / df['CME_Option_Strike_Price']

    # 修复并计算 30日 滚动波动率
    if 'Vol_Rolling_30D' not in df.columns or df['Vol_Rolling_30D'].isnull().all():
        df['Vol_Rolling_30D'] = df['Price_Return'].rolling(window=30).std() * np.sqrt(252)
    df['Vol_Rolling_30D'] = df['Vol_Rolling_30D'].fillna(df['Vol_Rolling_30D'].mean()).fillna(0)

    # 核心要求 2：GRU 专属特征工程严格复刻
    df['Log_Actual'] = np.log(df['Actual_CME_Price'] + 1e-6)
    df['Log_BSM'] = np.log(df['Paper_Original_BSM'] + 1e-6)
    df['Target_Log_Resid'] = df['Log_Actual'] - df['Log_BSM']
    df['VIX_Smooth'] = df['VIX_Volatility_Index'].rolling(5).mean().ffill().fillna(0)
    df['Moneyness_Log'] = np.log(df['Moneyness'] + 1e-6)

    return df.ffill().fillna(0)


# ==========================================
# 3. 数据加载与缓存
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        # 生成含有所需全量特征的模拟数据，以防未上传文件时崩溃
        dates = pd.date_range(start="2024-01-01", periods=500)
        jpm_price = np.random.normal(150, 5, 500)
        bsm_price = np.random.normal(15, 2, 500)
        noise = np.random.normal(0, 0.5, 500)
        df = pd.DataFrame({
            'Date': dates,
            'JPMorgan_Stock_Price': jpm_price,
            'CME_Option_Strike_Price': 150,
            'VIX_Volatility_Index': np.random.uniform(0.1, 0.4, 500),
            'News_Sentiment_MA7': np.random.uniform(-1, 1, 500),
            'Fed_Interest_Rate': np.random.uniform(0.01, 0.05, 500),
            'Paper_Original_BSM': bsm_price,
            'Actual_CME_Price': bsm_price * np.exp(noise * 0.1) + noise
        })
    return prepare_features(df)


# ==========================================
# 4. 模型一键训练模块 (修复时序对齐与 Scaler 域)
# ==========================================
@st.cache_resource
def train_models(df_full):
    models = {}
    scalers = {}

    # ==== 核心修复 1：精准计算时间序列截断点 ====
    time_steps = 4
    seq_len = len(df_full) - time_steps
    seq_split = int(seq_len * 0.8)
    test_start_idx = time_steps + seq_split  # 测试集在原 df_full 中的绝对起始索引

    models['test_start_idx'] = test_start_idx
    models['GRU_TimeSteps'] = time_steps

    # 静态模型的训练集严格卡在 test_start_idx 之前
    df_train = df_full.iloc[:test_start_idx].copy()

    # --- M0: BSM 代理模型 ---
    static_features = ['JPMorgan_Stock_Price', 'CME_Option_Strike_Price', 'VIX_Volatility_Index', 'News_Sentiment_MA7',
                       'Fed_Interest_Rate']
    gb_bsm = HistGradientBoostingRegressor(random_state=7)
    gb_bsm.fit(df_train[static_features], df_train['Paper_Original_BSM'])
    models['BSM_Proxy'] = gb_bsm
    models['Static_Feats'] = static_features

    # --- M1: Hybrid ---
    rf_vol = RandomForestRegressor(n_estimators=50, random_state=7)
    rf_vol.fit(df_train[static_features], df_train['Vol_Rolling_30D'])
    df_train_hybrid = df_train.copy()
    df_train_hybrid['Predicted_Vol'] = rf_vol.predict(df_train[static_features])
    hybrid_features = static_features + ['Predicted_Vol']

    gb_hybrid = HistGradientBoostingRegressor(random_state=7)
    gb_hybrid.fit(df_train_hybrid[hybrid_features], df_train['Actual_CME_Price'])
    models['Hybrid_Vol'] = rf_vol
    models['Hybrid_Price'] = gb_hybrid
    models['Hybrid_Feats'] = hybrid_features

    # --- M2: E2E Pricing ---
    gb_e2e = HistGradientBoostingRegressor(random_state=7)
    gb_e2e.fit(df_train[static_features], df_train['Actual_CME_Price'])
    models['E2E'] = gb_e2e

    # --- M3: LSTM ---
    lstm_time_steps = 10
    lstm_features = ['JPMorgan_Stock_Price', 'CME_Option_Strike_Price', 'VIX_Volatility_Index', 'Vol_Rolling_30D',
                     'News_Sentiment_MA7']
    scaler_lstm = StandardScaler()
    X_lstm_scaled = scaler_lstm.fit_transform(df_full[lstm_features])

    X_lstm_seq, y_lstm_seq = [], []
    for i in range(lstm_time_steps, len(X_lstm_scaled)):
        X_lstm_seq.append(X_lstm_scaled[i - lstm_time_steps:i])
        y_lstm_seq.append(df_full['Actual_CME_Price'].values[i])
    X_lstm_seq, y_lstm_seq = np.array(X_lstm_seq), np.array(y_lstm_seq)

    lstm_split = test_start_idx - lstm_time_steps
    model_lstm = Sequential([
        layers.Input(shape=(lstm_time_steps, len(lstm_features))),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_lstm_seq[:lstm_split], y_lstm_seq[:lstm_split], epochs=30, batch_size=32, verbose=0)

    models['LSTM'] = model_lstm
    scalers['LSTM'] = scaler_lstm
    models['LSTM_Feats'] = lstm_features
    models['LSTM_TimeSteps'] = lstm_time_steps

    # ====================================================
    # --- M4: GRU (严格复刻你的代码) ---
    # ====================================================
    # --- M4: GRU (严格复刻你的代码) ---
    # ==== 核心修复 3：强行重置种子，抹除前置模型对随机状态的消耗 ====
    # 直接调用全局库即可，千万不要在这里 import
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)
    # ====================================================

    gru_features = ['Moneyness_Log', 'VIX_Smooth', 'News_Sentiment_MA7', 'Price_Return', 'Log_BSM']
    scaler_gru = StandardScaler()
    # ... 后面的代码保持不变 ...


    X_gru_scaled = scaler_gru.fit_transform(df_full[gru_features])
    y_gru_full = df_full['Target_Log_Resid'].values

    X_gru_seq, y_gru_seq = [], []
    for i in range(time_steps, len(X_gru_scaled)):
        X_gru_seq.append(X_gru_scaled[i - time_steps:i])
        y_gru_seq.append(y_gru_full[i])

    X_gru_seq = np.array(X_gru_seq)
    y_gru_seq = np.array(y_gru_seq)

    # 精确切分训练集
    X_train_gru = X_gru_seq[:seq_split]
    y_train_gru = y_gru_seq[:seq_split]

    model_gru = Sequential([
        layers.Input(shape=(time_steps, len(gru_features))),
        layers.GRU(48, kernel_regularizer='l2'),
        layers.BatchNormalization(),
        layers.Dense(24, activation='relu'),
        layers.Dense(1)
    ])
    model_gru.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='huber')
    model_gru.fit(X_train_gru, y_train_gru, epochs=80, batch_size=32, verbose=0)

    # 偏差校准提取
    train_log_resid_preds = model_gru.predict(X_train_gru, verbose=0).flatten()
    bias_shift = np.mean(y_train_gru - train_log_resid_preds)

    models['GRU'] = model_gru
    scalers['GRU'] = scaler_gru
    models['GRU_Feats'] = gru_features
    models['GRU_Bias'] = bias_shift

    return models, scalers

# ==========================================
# 5. 预测生成引擎
# ==========================================
def generate_predictions(df_full, models, scalers, is_stress_test=False):
    """基于全量数据生成时序，截取验证集部分返回结果字典。如果是压力测试，需要动态重估 BSM"""
    split_idx = int(len(df_full) * 0.8)
    df_eval = df_full.copy()

    # 压力测试下：市场突变导致 BSM 理论价发生位移，利用代理模型重估
    if is_stress_test:
        df_eval['Paper_Original_BSM'] = models['BSM_Proxy'].predict(df_eval[models['Static_Feats']])
        # BSM 变化后，需要重算对数与残差目标，刷新依赖于 BSM 的底层特征
        df_eval = prepare_features(df_eval)

    df_test = df_eval.iloc[split_idx:].copy()
    preds = {'Date': df_test['Date'].values, 'Actual_CME_Price': df_test['Actual_CME_Price'].values,
             'Paper_Original_BSM': df_test['Paper_Original_BSM'].values}

    # E2E 预测
    preds['E2E'] = models['E2E'].predict(df_test[models['Static_Feats']])

    # Hybrid 预测
    pred_vol = models['Hybrid_Vol'].predict(df_test[models['Static_Feats']])
    df_test_hyb = df_test.copy()
    df_test_hyb['Predicted_Vol'] = pred_vol
    preds['Hybrid'] = models['Hybrid_Price'].predict(df_test_hyb[models['Hybrid_Feats']])

    # LSTM 预测
    lstm_scaled = scalers['LSTM'].transform(df_eval[models['LSTM_Feats']])
    lstm_seqs = np.array([lstm_scaled[i - 10:i] for i in range(split_idx, len(lstm_scaled))])
    preds['LSTM'] = models['LSTM'].predict(lstm_seqs, verbose=0).flatten()

    # GRU 预测与还原平滑 (严格执行)
    ts = models['GRU_TimeSteps']
    gru_scaled = scalers['GRU'].transform(df_eval[models['GRU_Feats']])
    gru_seqs = np.array([gru_scaled[i - ts:i] for i in range(split_idx, len(gru_scaled))])
    log_resid_preds = models['GRU'].predict(gru_seqs, verbose=0).flatten()

    log_bsm_test = df_test['Log_BSM'].values
    final_log_preds = log_bsm_test + log_resid_preds + models['GRU_Bias']
    ml_prices = np.exp(final_log_preds)
    preds['GRU'] = pd.Series(ml_prices).rolling(window=3, min_periods=1).mean().values

    return pd.DataFrame(preds)


# ==========================================
# 6. UI 构建与主程序
# ==========================================
st.sidebar.title("⚙️ 数据与参数面板")
uploaded_file = st.sidebar.file_uploader("📁 上传比较数据 (CSV)", type="csv")
df_master = load_data(uploaded_file)

# --- 核心要求 4：建议文件名与数据解说展开框 ---
st.title("🏦 JPM 期权定价分析：BSM vs 机器学习融合模型")

with st.expander("📝 建议文件名与数据解说", expanded=False):
    st.markdown("""
    | 字段名称 | 物理意义与用途解说 |
    | :--- | :--- |
    | **Date** | 交易日期，作为捕捉时间序列趋势的基准索引。 |
    | **Actual_CME_Price** | 合成市场真实价，由 T1(半年)平值 Put 与 T2(一年)平值 Call 组合而成。 |
    | **Paper_Original_BSM** | 基于论文参数及 Rubinstein (1991) 解析解计算出的 BSM 理论参考价。 |
    | **JPMorgan_Stock_Price** | JPMorgan (JPM) 股票每日收盘价。 |
    | **CME_Option_Strike_Price**| 期权行权价 (K)，定价公式的核心参数之一。 |
    | **VIX_Volatility_Index** | CBOE 波动率指数，反映市场恐慌程度与整体风险环境。 |
    | **News_Sentiment_MA7** | 7日金融新闻情绪移动平均值，用于修正非理性溢价驱动的定价偏差。 |
    | **Fed_Interest_Rate** | 无风险利率 (r)，反映资金的时间价值。 |
    | **Vol_Rolling_30D** | 标的资产收益率的30日滚动年化波动率 (σ)。 |
    """)

# 训练并预测
with st.spinner("🚀 正在固定随机种子并初始化神经网络引擎 (LSTM & GRU)..."):
    models, scalers = train_models(df_master)
    test_results_df = generate_predictions(df_master, models, scalers, is_stress_test=False)

# --- 模型综合性能看板 (降序排序) ---
st.subheader("📊 测试集性能看板 (严格按照 R² 排序)")
metrics_data = []
eval_models = ['Paper_Original_BSM', 'Hybrid', 'E2E', 'LSTM', 'GRU']

for model in eval_models:
    mae = mean_absolute_error(test_results_df['Actual_CME_Price'], test_results_df[model])
    r2 = r2_score(test_results_df['Actual_CME_Price'], test_results_df[model])
    metrics_data.append({"Model": model, "MAE (误差↓)": round(mae, 4), "R² (拟合度↑)": round(r2, 4)})

metrics_df = pd.DataFrame(metrics_data).set_index("Model")
metrics_df = metrics_df.sort_values(by="R² (拟合度↑)", ascending=False)  # 强制降序

st.dataframe(
    metrics_df.style.highlight_max(subset=['R² (拟合度↑)'], color='#2a4d69')
    .highlight_min(subset=['MAE (误差↓)'], color='#2a4d69'),
    use_container_width=True
)

st.markdown("---")

# --- 交互式可视化图表 ---
st.subheader("📈 测试集定价趋势与贴合度对比")
selected_models = st.multiselect(
    "👉 选择要在图表中显示的模型：",
    options=eval_models,
    default=['Paper_Original_BSM', 'GRU']
)

fig = go.Figure()
# 真实价格
fig.add_trace(go.Scatter(x=test_results_df['Date'], y=test_results_df['Actual_CME_Price'],
                         mode='lines', name='Actual CME Price',
                         line=dict(color=COLOR_MAP['Actual_CME_Price'], width=3)))

for model in selected_models:
    dash_style = 'dash' if model == 'Paper_Original_BSM' else 'solid'
    width_style = 2 if model == 'Paper_Original_BSM' else 1.5
    fig.add_trace(go.Scatter(x=test_results_df['Date'], y=test_results_df[model],
                             mode='lines', name=model,
                             line=dict(color=COLOR_MAP.get(model, '#888888'), width=width_style, dash=dash_style)))

fig.update_layout(
    height=450, template="plotly_dark", hovermode="x unified",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='#333333'),
    yaxis=dict(showgrid=True, gridcolor='#333333'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. 极端场景敏感性分析 (核心要求 5)
# ==========================================
st.markdown("---")
st.subheader("⚡ 极端场景敏感性分析 (Stress Test)")
st.markdown(
    "评估各模型在面临市场参数突变时的**价格重估均值**。底层 BSM 将动态位移，混合/端到端/深度网络将基于新时序特征重新输出。")

st.sidebar.markdown("### ⚡ 压力测试控制台")
vol_shock = st.sidebar.slider("VIX 与历史波动率飙升乘数 (倍)", 1.0, 3.0, 1.5, 0.1)
rate_shock = st.sidebar.slider("无风险利率骤增 (绝对百分点 %)", 0.0, 5.0, 2.0, 0.1)

if st.button("🚀 运行极端压力推演"):
    with st.spinner("正在重新生成时间序列并运算多模型场景..."):
        # Base 场景均值
        base_means = test_results_df[eval_models].mean().to_dict()

        # 场景 1：波动率飙升
        df_vol = df_master.copy()
        df_vol['VIX_Volatility_Index'] *= vol_shock
        df_vol['Vol_Rolling_30D'] *= vol_shock
        res_vol_df = generate_predictions(df_vol, models, scalers, is_stress_test=True)
        vol_means = res_vol_df[eval_models].mean().to_dict()

        # 场景 2：利率上升
        df_rate = df_master.copy()
        df_rate['Fed_Interest_Rate'] += (rate_shock / 100.0)  # 转为小数格式
        res_rate_df = generate_predictions(df_rate, models, scalers, is_stress_test=True)
        rate_means = res_rate_df[eval_models].mean().to_dict()

        # 构造对比数据
        stress_data = []
        for m in eval_models:
            stress_data.append({"Model": m, "Scenario": "Normal Base", "Avg Price": base_means[m]})
            stress_data.append({"Model": m, "Scenario": f"Vol Shock ({vol_shock}x)", "Avg Price": vol_means[m]})
            stress_data.append({"Model": m, "Scenario": f"Rate Hike (+{rate_shock}%)", "Avg Price": rate_means[m]})

        stress_df = pd.DataFrame(stress_data)

        # 绘制分组柱状图
        fig_stress = px.bar(stress_df, x='Model', y='Avg Price', color='Scenario', barmode='group',
                            color_discrete_sequence=['#4daf4a', '#e41a1c', '#377eb8'])

        fig_stress.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=400, yaxis_title="Average Predicted Price ($)",
            legend_title="Market Environment"
        )
        st.plotly_chart(fig_stress, use_container_width=True)
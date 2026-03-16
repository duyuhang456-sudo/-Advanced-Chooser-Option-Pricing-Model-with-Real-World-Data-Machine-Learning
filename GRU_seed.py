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
    X_lstm_scaled = scaler_lstm.fit_transform(df_full[lstm_features])  # 必须针对全量数据做缩放以消除断层

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

    # --- M4: GRU (严格复刻你的代码) ---
    gru_features = ['Moneyness_Log', 'VIX_Smooth', 'News_Sentiment_MA7', 'Price_Return', 'Log_BSM']
    scaler_gru = StandardScaler()

    # ==== 核心修复 2：在全量特征上 fit_transform ====
    X_gru_scaled = scaler_gru.fit_transform(df_full[gru_features])
    y_gru_full = df_full['Target_Log_Resid'].values

    X_gru_seq, y_gru_seq = [], []
    for i in range(time_steps, len(X_gru_scaled)):
        X_gru_seq.append(X_gru_scaled[i - time_steps:i])
        y_gru_seq.append(y_gru_full[i])

    X_gru_seq = np.array(X_gru_seq)
    y_gru_seq = np.array(y_gru_seq)

    # 精确切分训练集 (长度与 seq_split 完全对应)
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
# 5. 预测生成引擎 (修复索引对齐映射)
# ==========================================
def generate_predictions(df_full, models, scalers, is_stress_test=False):
    test_start_idx = models['test_start_idx']
    df_eval = df_full.copy()

    if is_stress_test:
        df_eval['Paper_Original_BSM'] = models['BSM_Proxy'].predict(df_eval[models['Static_Feats']])
        df_eval = prepare_features(df_eval)

    # ==== 确保测试集的 Date 与 预测出的价格 100% 对齐 ====
    df_test = df_eval.iloc[test_start_idx:].copy()
    preds = {
        'Date': df_test['Date'].values,
        'Actual_CME_Price': df_test['Actual_CME_Price'].values,
        'Paper_Original_BSM': df_test['Paper_Original_BSM'].values
    }

    # E2E 预测
    preds['E2E'] = models['E2E'].predict(df_test[models['Static_Feats']])

    # Hybrid 预测
    df_test_hyb = df_test.copy()
    df_test_hyb['Predicted_Vol'] = models['Hybrid_Vol'].predict(df_test[models['Static_Feats']])
    preds['Hybrid'] = models['Hybrid_Price'].predict(df_test_hyb[models['Hybrid_Feats']])

    # LSTM 预测
    lstm_ts = models['LSTM_TimeSteps']
    lstm_scaled = scalers['LSTM'].transform(df_eval[models['LSTM_Feats']])
    lstm_seqs = np.array([lstm_scaled[i - lstm_ts:i] for i in range(test_start_idx, len(lstm_scaled))])
    preds['LSTM'] = models['LSTM'].predict(lstm_seqs, verbose=0).flatten()

    # GRU 预测与还原平滑
    gru_ts = models['GRU_TimeSteps']
    gru_scaled = scalers['GRU'].transform(df_eval[models['GRU_Feats']])
    gru_seqs = np.array([gru_scaled[i - gru_ts:i] for i in range(test_start_idx, len(gru_scaled))])

    log_resid_preds = models['GRU'].predict(gru_seqs, verbose=0).flatten()
    log_bsm_test = df_test['Log_BSM'].values

    final_log_preds = log_bsm_test + log_resid_preds + models['GRU_Bias']
    ml_prices = np.exp(final_log_preds)
    preds['GRU'] = pd.Series(ml_prices).rolling(window=3, min_periods=1).mean().values

    return pd.DataFrame(preds)
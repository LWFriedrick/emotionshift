# B2 BiGRU / BiLSTM results

Mean +/- population std across seeds; CI is 95% dialogue bootstrap of the first seed.
DailyDialog headline: test_dedup. Architecture selection uses dev only.

| Dataset | Model | Dev selected | Split | n | Weighted F1 | Macro F1 | Micro F1 excl. neutral |
|---|---|---|---|---|---|---|---|
| meld | b2_bigru | True | test | 2610 | 0.5508 +/- 0.0045 [0.5244, 0.5673] | 0.3538 +/- 0.0095 [0.3284, 0.3744] | 0.3876 +/- 0.0067 [0.3596, 0.4178] |
| meld | b2_bilstm | False | test | 2610 | 0.5490 +/- 0.0043 [0.5261, 0.5712] | 0.3403 +/- 0.0049 [0.3148, 0.3546] | 0.3823 +/- 0.0059 [0.3493, 0.4033] |
| dailydialog | b2_bigru | False | test_dedup | 7019 | 0.8175 +/- 0.0028 [0.8060, 0.8314] | 0.4065 +/- 0.0155 [0.3666, 0.4563] | 0.4895 +/- 0.0051 [0.4682, 0.5210] |
| dailydialog | b2_bigru | False | test | 7740 | 0.8152 +/- 0.0030 [0.8055, 0.8297] | 0.4210 +/- 0.0151 [0.3818, 0.4723] | 0.4917 +/- 0.0078 [0.4702, 0.5259] |
| dailydialog | b2_bilstm | True | test_dedup | 7019 | 0.8183 +/- 0.0026 [0.8043, 0.8300] | 0.3883 +/- 0.0287 [0.3683, 0.4564] | 0.4828 +/- 0.0078 [0.4604, 0.5136] |
| dailydialog | b2_bilstm | True | test | 7740 | 0.8158 +/- 0.0022 [0.8026, 0.8271] | 0.4018 +/- 0.0276 [0.3797, 0.4681] | 0.4841 +/- 0.0079 [0.4621, 0.5152] |

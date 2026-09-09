# フェーズ 1: セットアップ（完了）

SO-101 のリーダー機・フォロワー機を接続し、キャリブレーションしてテレオペレーションまで通す手順です。

## 動作確認済み環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 Home（WSL の bash / PowerShell どちらでも可） |
| Python | 3.11（uv が自動で用意） |
| LeRobot | 0.4.4 |
| USB シリアル | WCH CH343（VID:PID = `1A86:55D3`）× 2 |
| サーボ | Feetech STS3215 × 12（リーダー 6 + フォロワー 6）、1,000,000 baud |

Docker は使っていません。Windows の Docker Desktop は USB/IP ブリッジなしに COM ポートを渡せないためです。

## 1. 接続確認

```bash
uv run scripts/ports.py         # 認識されているシリアルポート一覧
uv run scripts/scan_servos.py   # 各ポートのサーボを ping
```

両アームで ID 1〜6 が応答すれば正常です。ボーレートは 1,000,000 / 500,000 / 115,200 の順に自動判定します。

```bash
uv run scripts/diag.py          # 角度・可動域・トルク・負荷・電圧・温度
```

電圧は 11〜13V、温度は 40℃ 前後が正常です。

## 2. 現在の設定をバックアップ

**キャリブレーション前に必ず実行してください。**

```bash
uv run scripts/check_stored_cal.py save
```

LeRobot の `set_half_turn_homings` は内部で `reset_calibration()` を呼び、サーボ EEPROM 内の `Min/Max_Position_Limit` と `Homing_Offset` を消去します。つまり**キャリブレーションに失敗すると、実行前より状態が悪くなります**。引数なしで再実行すると、保存時点との差分が表示されます。

## 3. リーダー機とフォロワー機の判別

```bash
uv run scripts/monitor.py
```

8 秒の準備時間の後、45 秒間すべての関節を監視します。この間に**リーダー機（手で操作する方）だけを手で大きく動かしてください。** 動いた側がリーダー、静止していた側がフォロワーと判定されます。

COM 番号は USB ポートを挿し替えると入れ替わるため、テレオペの動きが逆になった場合はこれを再実行してください。

## 4. 駆動テスト（任意）

```bash
uv run scripts/move_test.py COM4 --amplitude-deg 25
```

1 関節ずつ 1.5 度刻みで往復させます。負荷が 800（フルスケール 1023）または温度が 60℃ を超えると中断します。トルクは `finally` で必ず解除されるため、中断してもアームは脱力状態で残ります。

安全のためポートの明示指定が必須です。**周囲に人や物がない状態で実行してください。**

## 5. キャリブレーション

```bash
uv run scripts/easy_calibrate.py --robot.type=so101_follower --robot.port=COM4 --robot.id=follower
uv run scripts/easy_calibrate.py --teleop.type=so101_leader  --teleop.port=COM3 --teleop.id=leader
```

`wrist_roll` 以外の全関節を、**順番も開始姿勢も気にせず**両端まで動かして ENTER を押すだけです。TRAVEL 列が数百 tick を超えていれば十分で、動かし足りない関節には `<- not moved yet` と表示されます。

出力は LeRobot 公式と同じ形式・同じ保存先です。

```
~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json
```

**このファイルはリポジトリに含まれません。** 新しい環境ではキャリブレーションからやり直しが必要です。同じアームを使う場合は、この 2 ファイルをコピーすれば再利用できます。

## 6. テレオペレーション

```bash
./scripts/teleop.sh
```

リーダー機を手で動かすとフォロワー機が追従します。停止は Ctrl+C です。

**開始前に、リーダー機をフォロワー機と近い姿勢に合わせてください。** 姿勢が離れているほど起動直後に大きく動きます。初回や不安な場合は 1 ステップあたりの移動量を制限できます。

```bash
./scripts/teleop.sh --robot.max_relative_target=5   # 1 ステップ 5 度に制限
```

ポートや設定は環境変数で上書きできます。

```bash
FOLLOWER_PORT=COM5 LEADER_PORT=COM6 FPS=60 ./scripts/teleop.sh
```

## LeRobot 公式コマンドをそのまま使わない理由

`src/so101/hardware/bus_patch.py` で 2 つの問題を回避しています。

### 1. `ConnectionError: Incorrect status packet!`

`record_ranges_of_motion` も 60 Hz のテレオペループも `sync_read` を `num_retry=0` で呼びます。Feetech のシンクリードは全サーボが連続応答する方式で、USB シリアル経由では時折パケットが化けます。リトライがないため、**1 回の化けで実行全体が例外終了**します。

`bus_patch.py` は読み書きを 10 回までリトライさせます。

### 2. `ValueError: Negative values are not allowed`

LeRobot のキャリブレーションは、**可動域を記録する前に**開始姿勢をエンコーダ値 2047 に固定します（`set_half_turn_homings`）。そのため開始姿勢が機械端に近い関節では、反対側の端が 0〜4095 の範囲外になり、記録が終わった最後の保存時に失敗します。45 秒間の記録がすべて失われます。

`easy_calibrate.py` は順序を逆にしました。**先に可動域を記録し、その中点からホーミングオフセットを逆算**します。開始姿勢は一切問わず、結果は常に 2047 中心・範囲内に収まります。

```
homing_offset = (min + max) / 2 - 2047
range_min     = min - homing_offset
range_max     = max - homing_offset
```

ENTER 直後に生データを `last_recording.json` へ保存するため、以降で何が起きても記録は残ります。

## 追従ラグの調整

**シリアル通信は律速ではありません。** `bench_latency.py` の実測で、6 サーボの sync_read は中央値 1.00ms（p95 1.06ms）です。読み書き合わせた 1 制御ステップは約 2ms で、60 Hz の周期 16.7ms の 12% に過ぎません。

効果の大きい順に 3 つあります。

| 調整項目 | 効果 |
|---|---|
| `--fps=120` | 制御周期が 16.7ms → 8.3ms に半減。バスに余裕があるため確実に効く。`teleop.sh` の既定値 |
| `--robot.max_relative_target` を外す | 1 ステップあたりの移動量上限を撤廃（`=5` なら 5 度 = 300 度/秒） |
| `P_Coefficient` | LeRobot は接続のたびにフォロワーへ 16 を書き込む（サーボ既定は 32、「振動を避けるため」）。`teleoperate.py` は 24 に設定。上げるほど追従が速くなるが、上げすぎると振動する |

```bash
SO101_P_COEFFICIENT=32 ./scripts/teleop.sh   # サーボ既定値。振動したら下げる
```

ただしラグはゼロにはなりません。サーボは有限ゲインの位置制御アクチュエータであり、フォロワーは自身の質量を加速させる必要があります。

## 補足

`Present_Load` は 11 ビット値です。bit 10 が方向、bits 0〜9 が大きさを表します。16 ビット符号付きとして読むと 1024 以上の偽の値になります。

`Homing_Offset` も同様に bit 11 が符号ビットです。

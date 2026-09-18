# SO-101

SO-101 ロボットアームの展示デモです。画面で色を選ぶと、カメラがその色のブロックを
探し、アームがそれを取って缶に入れます。

ハードウェアの立ち上げ、カメラ、ブロック検出、そして展示用の画面までを 1 つの
リポジトリで扱います。

## ロードマップ

| フェーズ | 内容 | 状態 | ドキュメント |
|---|---|---|---|
| 1 | **セットアップ** — 接続確認、キャリブレーション、テレオペレーション | ✅ 完了 | [docs/01-setup.md](docs/01-setup.md) |
| 2 | **カメラ認識** — カメラ検出、同期、映像取得 | ✅ 完了 | [docs/02-camera.md](docs/02-camera.md) |
| 3 | **物体検出** — 側面カメラでの色別ブロック検出（背景色に非依存） | ✅ 完了 | [docs/07-vision.md](docs/07-vision.md) |
| 4 | **テレオペのラグ調整** — 追従遅れ・速度差・表示レートの実測と対処 | ✅ 完了 | [docs/06-teleop-tuning.md](docs/06-teleop-tuning.md) |
| 5 | **展示デモ（Slot 方式）** — 色を指定してブロックを取り、缶へ入れる | ✅ 完了 | [docs/08-demo.md](docs/08-demo.md) |
| 6 | **画面** — コマンドを使わずに 2 つのモードを操作する | ✅ 完了 | [docs/05-app.md](docs/05-app.md) |

### 以前あったもの

このリポジトリには一時期、模倣学習（実演の記録・ACT ポリシーの学習と推論）と、
座標変換にもとづく自律把持（URDF からの逆運動学、ピクセル→テーブルのホモグラフィ、
TCP モデル、ビジュアルサーボ、MuJoCo のデジタルツイン）が入っていました。
展示デモはそのどれも使わないため、いまは削除してあります。**Git の履歴には
すべて残っています**（`555d224` 以前）。

### 展示デモの考え方

**座標を一切使いません。** 色 → 検出 → **ピクセル空間**で最も近い Slot →
その Slot について人が教示した関節角の再生、という流れです。逆運動学も
カメラ→ロボットの変換も通しません。

- **側面カメラ（外付け）** — どの色がどこにあるかだけを答えます。答えはピクセルです
- **アーム搭載カメラ（手首）** — 画面に映すためのもので、判断には使いません

これは制約であって手抜きではありません。会場で合わせ直すのは Slot の位置
（クリック 6 回）だけで、較正がずれて腕が机を叩くという失敗の仕方をしません。
ブロックがどの Slot からも遠ければ、動かずに拒否します。

## セットアップ

[uv](https://docs.astral.sh/uv/) が必要です。未インストールなら PowerShell で以下を実行してください。

```powershell
winget install --id=astral-sh.uv -e
```

リポジトリを取得して環境を構築します。Python 3.11 と LeRobot 一式（PyTorch 含む、約 3GB）が入ります。

```bash
git clone https://github.com/t-seino-ml/SO-101.git
cd SO-101
uv sync
```

`.venv` が 1 つ作られ、これで全スクリプトが動きます。pyserial は LeRobot の依存に含まれるため、追加の環境は不要です。

## 動作確認（フェーズ 1）

```bash
uv run scripts/ports.py            # シリアルポートの検出
uv run scripts/scan_servos.py      # サーボの ping
uv run scripts/diag.py             # 電圧・温度・可動域などの健全性診断
uv run scripts/find_cameras.py     # カメラの検出とプレビュー保存
```

キャリブレーションとテレオペレーションの手順は [docs/01-setup.md](docs/01-setup.md) を参照してください。

```bash
./scripts/teleop.sh                # 2 台のカメラ映像を見ながらテレオペ
```

rerun のビューアが開き、`overhead` と `side` の映像と関節角度が表示されます。制御は 200 Hz、表示は 30 Hz です。初回は `--max-relative-target 5` を付けると 1 ステップ 5 度に制限できます。

**WSL から `uv run` を使わないでください。** WSL の `uv` は Linux 版で、Windows の `.venv` を消して Linux 用に作り直します。COM ポートもカメラも見えなくなります。`teleop.sh` は Windows の Python を直接呼ぶので WSL から安全に使えます。`uv run` を使う場合は PowerShell から実行してください。全スクリプトが起動時にこれを検査し、誤った環境なら即座に停止します。

## 構成

```
SO-101/
├── docs/                  フェーズごとのドキュメント
├── src/so101/
│   ├── hardware/          サーボバス、ポート検出、LeRobot パッチ（実装済み）
│   ├── camera/            カメラ検出・取得（実装済み）
│   ├── teleop.py          制御と表示を分離したテレオペループ（実装済み）
│   ├── dataset/           記録・データセット化（実装済み）
│   ├── policy/            ブロック検出、運動、把持計画（実装済み）
│   ├── demo/              展示デモの軌道と Slot 判定（実装済み）
│   ├── sim/               MuJoCo デジタルツイン（実装済み）
│   └── ui/                画面部品（プレースホルダ、未実装）
├── scripts/               CLI ツール
├── app/
│   ├── demo_ui.py         展示用の画面。2 モードの入口
│   └── theme.py           画面の描画部品（角丸・グラデーション・立方体）
└── tests/
```

空のプレースホルダは `src/so101/ui/` だけです。フレームワークに依存しない画面部品を
置く想定でしたが、展示用の画面は Tkinter を直に使う `app/` 側に入っています。

カメラの役割（どれが俯瞰か）は物理的な配置の問題で自動検出できないため、リポジトリ直下の `cameras.json` に記述します。**別のリグではこのファイルを編集してください。**

## スクリプト一覧

| ファイル | 用途 |
|---|---|
| `scripts/ports.py` | シリアルポートの列挙と自動検出。CH34x を優先 |
| `scripts/scan_servos.py` | 各バスのサーボを ping し、ボーレートを自動判定 |
| `scripts/diag.py` | 全サーボの角度・可動域・トルク・負荷・電圧・温度を表示 |
| `scripts/monitor.py` | 手動操作を検出し、リーダー機とフォロワー機を判別 |
| `scripts/check_stored_cal.py` | サーボ内蔵キャリブレーションの表示・保存・差分確認 |
| `scripts/move_test.py` | 1 関節ずつ往復させる駆動テスト。負荷・温度で自動中断 |
| `scripts/bench_latency.py` | バスの sync_read レイテンシを実測 |
| `scripts/easy_calibrate.py` | 中間姿勢の指定が不要なキャリブレーション |
| `scripts/teleoperate.py` | テレオペ本体。通信リトライと P ゲイン調整を追加 |
| `scripts/teleop.sh` | 検証済み設定でテレオペを起動するランチャー |
| `scripts/find_cameras.py` | カメラの列挙と、1 台ごとのプレビュー画像保存 |
| `scripts/bench_camera.py` | カメラのフレームレートと、制御ループへの影響を実測 |
| `scripts/teleop_view.py` | カメラ映像付きテレオペ。`--check` で接続確認のみ |
| `scripts/bench_teleop.py` | 表示レートと安全クランプが制御ループに与える影響を実測 |
| `scripts/clear_overload.py` | 過負荷でラッチしたサーボの保護状態を解除 |

引数なしで実行するとポートを自動検出します。`move_test.py` のみ、安全のためポートの明示指定が必須です。

表は接続・診断まわりだけです。ブロック検出器を作り直すときは次を使います。
手順は [docs/07-vision.md](docs/07-vision.md) にあります。

| ファイル | 用途 |
|---|---|
| `scripts/extract_blocks.py` | 写真からブロックを切り抜く（アルファ付き） |
| `scripts/capture_backgrounds.py` | 実写の背景を撮る。アームも一緒に写す |
| `scripts/build_dataset.py` | 切り抜きと背景から YOLO 用の合成データセットを生成 |
| `scripts/train_detector.py` | 検出器を学習 |
| `scripts/eval_detector.py` | 学習した検出器を実写で評価 |

展示デモで使うものは次の節にまとめてあります。

## 展示デモを動かす

画面から操作します。コマンドは 1 つだけです。

```bash
uv run app/demo_ui.py              # F11 で全画面、Esc で戻る
```

ホーム画面で 2 つのモードを選べます。

- **リーダ機で動かす** — もう一方のアームを手で動かすと、ロボットが同じ形について来ます。
  モデルも検出も使いません
- **ブロックをつかむ** — 色を選ぶと、側面カメラがその色を探し、いちばん近い Slot の
  教示済み軌道を再生してブロックを缶へ入れます

色 → YOLO 検出 → **ピクセル空間**で最も近い Slot → 教示した関節角の再生、という流れです。
座標変換も IK も通しません。ブロックがどの Slot からも遠ければ、動かずに拒否します。

| ファイル | 用途 |
|---|---|
| `app/demo_ui.py` | 展示用の画面。通常はこれだけを起動します |
| `scripts/teach_demo_slot.py` | リーダ機で 1 Slot 分の軌道を教示して保存 |
| `scripts/calibrate_demo_slots.py` | 映像上で Slot の中心をクリックして登録 |
| `scripts/demo_slot_pick.py` | 画面を使わず CLI から実行。`--slot N` は検出を介さない最終手段 |
| `scripts/torque_off.py` | **緊急停止。** その場で保持してトルクを切ります |

会場での手順、拒否の条件、止め方の 3 系統は [docs/08-demo.md](docs/08-demo.md) にあります。

> **教示データはこのリポジトリに入っていません。** `data/` は `.gitignore` の対象で、
> `data/demo_slots/slot1〜6.json`（教示した軌道）と `data/demo_slot_calibration.json`
> （Slot の位置）は各機体のキャリブレーションに固有です。別の機体では
> `teach_demo_slot.py` と `calibrate_demo_slots.py` で作り直してください。

> **緊急停止は物理です。** 画面の STOP は動作の区切りで止まります（最大 3 秒）。
> すぐ止めるときは電源、または別ターミナルで `uv run scripts/torque_off.py COM4`。


## テスト

実機を必要としないロジックのテストです。

```bash
uv run pytest
```

実機が必要な確認は `scripts/` 配下のツールで行います。

## 動作確認済み環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 Home（WSL の bash / PowerShell どちらでも可） |
| Python | 3.11（uv が自動で用意） |
| LeRobot | 0.4.4 |
| USB シリアル | WCH CH343（VID:PID = `1A86:55D3`）× 2 |
| サーボ | Feetech STS3215 × 12、1,000,000 baud |
| カメラ | UVC カメラ（DirectShow）。1280x720 MJPG で約 20 fps |

Docker は使っていません。Windows の Docker Desktop は USB/IP ブリッジなしに COM ポートを渡せないためです。

## LeRobot 公式コマンドをそのまま使わない理由

`src/so101/hardware/bus_patch.py` で 2 つの問題を回避しています。どちらもフェーズ 3 以降（`lerobot-record`、推論ループ）でも同じように効いてくる想定です。

1. **`sync_read` にリトライがない** — キャリブレーションの記録ループもテレオペの制御ループも `num_retry=0` で呼ぶため、USB シリアル経由でパケットが 1 回化けただけで実行全体が例外終了します
2. **可動域を知る前に中間姿勢を固定する** — 開始姿勢が機械端に近いと、最後の保存時に範囲外エラーで記録がすべて失われます。`easy_calibrate.py` は順序を逆にして回避しています
3. **パケットタイムアウトに固定 +50ms** — パケットが 1 回落ちるたびに制御が 50〜100ms 停止します。5ms に下げることで、10 秒間に 20 件あった 20ms 超の停止が 0 件になりました

詳細は [docs/01-setup.md](docs/01-setup.md) を参照してください。

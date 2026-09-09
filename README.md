# SO-101

SO-101 ロボットアームを使った模倣学習アプリケーションの構築プロジェクトです。

ハードウェアのセットアップから、カメラ認識、データ記録、モデル学習、UI、そしてそれらを統合したアプリケーションまでを 1 つのリポジトリで扱います。

## ロードマップ

| フェーズ | 内容 | 状態 | ドキュメント |
|---|---|---|---|
| 1 | **セットアップ** — 接続確認、キャリブレーション、テレオペレーション | ✅ 完了 | [docs/01-setup.md](docs/01-setup.md) |
| 2 | **カメラ認識** — カメラ検出、同期、映像取得 | 未着手 | [docs/02-camera.md](docs/02-camera.md) |
| 3 | **データセット記録** — 操作の記録、LeRobot データセット化 | 未着手 | [docs/03-dataset.md](docs/03-dataset.md) |
| 4 | **モデル構築・推論** — ポリシー学習、実機での自律動作 | 未着手 | [docs/04-training.md](docs/04-training.md) |
| 5 | **UI・統合アプリ** — 全体を 1 つのアプリケーションに統合 | 未着手 | [docs/05-app.md](docs/05-app.md) |

フェーズ 2 以降のドキュメントは**計画**であり、実装・検証は行っていません。

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
```

キャリブレーションとテレオペレーションの手順は [docs/01-setup.md](docs/01-setup.md) を参照してください。

```bash
./scripts/teleop.sh                # テレオペレーション開始
```

## 構成

```
SO-101/
├── docs/                  フェーズごとのドキュメント
├── src/so101/
│   ├── hardware/          サーボバス、ポート検出、LeRobot パッチ（実装済み）
│   ├── camera/            カメラ検出・取得（フェーズ 2）
│   ├── dataset/           記録・データセット化（フェーズ 3）
│   ├── policy/            学習・推論（フェーズ 4）
│   └── ui/                画面部品（フェーズ 5）
├── scripts/               CLI ツール
├── app/                   統合アプリケーション（フェーズ 5）
└── tests/
```

`src/so101/` 配下の `camera` / `dataset` / `policy` / `ui` は、各パッケージの `__init__.py` に想定する役割だけを記載したプレースホルダです。実装はまだありません。

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
| `scripts/calibrate.py` | LeRobot 公式キャリブレーション（通信リトライのみ追加） |
| `scripts/teleoperate.py` | テレオペ本体。通信リトライと P ゲイン調整を追加 |
| `scripts/teleop.sh` | 検証済み設定でテレオペを起動するランチャー |

引数なしで実行するとポートを自動検出します。`move_test.py` のみ、安全のためポートの明示指定が必須です。

## 動作確認済み環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 Home（WSL の bash / PowerShell どちらでも可） |
| Python | 3.11（uv が自動で用意） |
| LeRobot | 0.4.4 |
| USB シリアル | WCH CH343（VID:PID = `1A86:55D3`）× 2 |
| サーボ | Feetech STS3215 × 12、1,000,000 baud |

Docker は使っていません。Windows の Docker Desktop は USB/IP ブリッジなしに COM ポートを渡せないためです。

## LeRobot 公式コマンドをそのまま使わない理由

`src/so101/hardware/bus_patch.py` で 2 つの問題を回避しています。どちらもフェーズ 3 以降（`lerobot-record`、推論ループ）でも同じように効いてくる想定です。

1. **`sync_read` にリトライがない** — キャリブレーションの記録ループもテレオペの制御ループも `num_retry=0` で呼ぶため、USB シリアル経由でパケットが 1 回化けただけで実行全体が例外終了します
2. **可動域を知る前に中間姿勢を固定する** — 開始姿勢が機械端に近いと、最後の保存時に範囲外エラーで記録がすべて失われます。`easy_calibrate.py` は順序を逆にして回避しています

詳細は [docs/01-setup.md](docs/01-setup.md) を参照してください。

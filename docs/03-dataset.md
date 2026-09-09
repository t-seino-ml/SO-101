# フェーズ 3: データセット記録（未着手）

> このドキュメントは**計画**です。実装・検証はまだ行っていません。

## 目的

テレオペレーションでの操作を記録し、学習に使える LeRobot データセット形式にします。

## LeRobot 側の既存機能

以下の CLI が `.venv` に入っています。自前で実装せず、これらをこのリグのポート・キャリブレーションに合わせて包むのが方針です。

| コマンド | 用途 |
|---|---|
| `lerobot-record` | テレオペしながらエピソードを記録 |
| `lerobot-dataset-viz` | 記録したデータセットの確認 |
| `lerobot-edit-dataset` | エピソードの削除・編集 |
| `lerobot-replay` | 記録した動作をフォロワー機で再生 |

## 注意点

`lerobot-record` も内部で `sync_read` を使うため、**フェーズ 1 と同じ `ConnectionError` が出る可能性が高い**です。`scripts/teleoperate.py` と同様に `so101.hardware.bus_patch` を import してから `lerobot.scripts.lerobot_record` の `main()` を呼ぶラッパーを用意することになります。

```python
from so101.hardware import bus_patch  # noqa: F401
from lerobot.scripts.lerobot_record import main
main()
```

## 決めること

- **タスク定義** — 何をさせるか。ピック&プレース、積み上げ、など
- **エピソード数** — 模倣学習では 50 エピソード程度が目安とされますが、タスクの難易度次第です
- **記録フレームレート** — 学習側の想定と揃える必要があります（30 fps が一般的）
- **初期姿勢の再現性** — エピソードごとに開始位置がばらつくと学習が難しくなります

## 確認したい項目

- [ ] `lerobot-record` がこの環境で通るか（リトライパッチの要否）
- [ ] 1 エピソードあたりのディスク使用量
- [ ] 記録中に制御ループのフレームレートが維持されるか

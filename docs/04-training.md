# フェーズ 4: ポリシー学習・実行

`data/demos` の実演から ACT ポリシーを学習し、フォロワー機で自律実行します。

## 学習

```powershell
uv run scripts/train_policy.py --steps 20000    # まず短縮版で効果を確認
uv run scripts/train_policy.py                  # 本番 100k ステップ
uv run scripts/train_policy.py --cameras wrist,side
```

入力は `observation.images.wrist` + `observation.state`、出力は 6 関節の目標角度。
色も座標も渡していません。

### なぜ `lerobot-train` を直接呼ばないか

LeRobot はデータセット内の全特徴からポリシーの入力を決めるため、両方の
カメラを記録したデータセットでは両方が入力になります。これを絞るには
`input_features` をポリシー構築**前**に設定する必要があり（`make_policy` は
空のときしか埋めません）、CLI からは届きません。`scripts/train_policy.py` は
`draccus.parse` で設定を作ってから `input_features` を差し替えています。

### Windows での注意

LeRobot は最新チェックポイントをシンボリックリンクで指しますが、Windows は
管理者権限なしにこれを作れません。保存**後**に失敗するのでチェックポイント
自体は無事ですが、例外で学習が止まります。`train_policy.py` は
`lerobot_train.update_last_checkpoint` を差し替え、代わりに `last.txt` に
名前を書きます（`train_utils` 側を差し替えても効きません。関数が直接
import されているためです）。

### 実測

RTX 4060 Laptop で約 3.0 step/s（batch 8、chunk 100）。

| ステップ数 | 所要時間の目安 |
|---|---|
| 20,000 | 約 1.8 時間 |
| 100,000 | 約 9 時間 |

学習曲線は `uv run scripts/watch_training.py` でリアルタイムに見られます。

## 実行

```powershell
uv run scripts/run_policy.py
uv run scripts/run_policy.py --seconds 20 --max-step 3
uv run scripts/run_policy.py --checkpoint runs/policy/grasp/checkpoints/020000
```

`last.txt` から最新チェックポイントを読み、ポリシーが宣言しているカメラだけを
開き、フォロワー機を 30 Hz で駆動します。

## 安全面（重要）

**自律動作ではリーダー機を人間が握っていません。** 以下を `run_policy.py` に
組み込んであります。

- 1 制御ステップあたりの関節移動量を `--max-step`（既定 4 度）に制限。
  予測が外れてもアームが飛びません
- 全関節の負荷・温度を毎ステップ監視し、閾値超過で中断
- Ctrl-C・時間切れ・過負荷のいずれでも、開始時の姿勢に戻してからトルクを抜く

物理的な非常停止手段（電源スイッチを手の届く位置に）は変わらず必要です。

## この先

学習済みポリシーは「目の前のブロックをつかむ」しかしません。フェーズ 5 で
統合します。

1. UI で色を選ぶ
2. 側面カメラの検出器がその色のブロックを見つける（`docs/07-vision.md`）
3. キャリブレーション済みのホモグラフィでアーム座標に変換し、粗く移動する
4. ここからポリシーに引き渡す

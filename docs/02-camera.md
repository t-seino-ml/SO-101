# フェーズ 2: カメラ認識（完了）

フォロワー機の作業領域を撮影し、後段のデータ記録・学習・推論に供給できる状態にしました。

## この環境で認識されたカメラ

```bash
uv run scripts/find_cameras.py
```

| index | デバイス名 | 内容 | 用途 |
|---|---|---|---|
| 0 | USB2.0 FHD UVC WebCam | ノート内蔵 | 使わない |
| 1 | **icspring camera** | 作業領域の俯瞰 | **overhead** |
| 2 | EMEET SmartCam S600 | 真っ黒（後述） | 未使用 |
| 3 | Camera (NVIDIA Broadcast) | 仮想カメラ | 使わない |

`find_cameras.py` は各カメラから 1 フレーム取得して `outputs/camera_previews/` に保存します。**画像を開けばどれがどのカメラか一目で分かります。**

### index 2（EMEET）について

フレームは届いていますが、輝度が平均 2/255・最大 27 で真っ黒です。露出の立ち上がり待ちを 40 フレームに延ばしても改善せず、むしろ暗くなりました（自動露出が暗いシーンに収束しているため）。

**レンズカバーが閉じているか、暗い方向を向いています。** 手先カメラとして使う場合はカバーを開けて `find_cameras.py` を再実行してください。

## インデックスではなく名前で識別する

OpenCV はカメラをインデックスで指しますが、Windows は抜き差しや列挙順でインデックスを付け替えます。シリアルポートで `hardware/ports.py` が解決しているのと同じ問題です。

DirectShow はデバイス名を提供し、**その列挙順は OpenCV の `CAP_DSHOW` インデックスと一致します**。そこで `camera/discovery.py` は名前で識別し、インデックスは開く直前に解決します。

```python
from so101.camera import resolve, open_camera
resolve("icspring")            # -> 現在のインデックス
open_camera("icspring")        # 名前でそのまま開ける
```

名前は部分一致・大文字小文字を無視します。設定に `"icspring"` と書いておけば、番号が変わっても動きます。

名前の取得には `pygrabber`（Windows のみ）を使います。利用できない環境では空リストを返し、インデックス指定にフォールバックします。

## フォーマットとフレームレート

DirectShow の既定は YUY2（非圧縮）で、USB 2.0 の帯域に律速されます。**MJPG を要求する場合は、解像度より先に FOURCC を設定する必要があります。** DirectShow はストリームフォーマットを先に決めるためです。

実測（icspring camera）:

| 要求 | 実際 | fps |
|---|---|---|
| 既定 640x480 | YUY2 640x480 | 19.7 |
| MJPG 640x480 | YUY2 640x480（要求が無視された） | 19.9 |
| MJPG 1280x720 | **MJPG 1280x720** | 20.0 |

このカメラは約 20 fps が上限です。解像度を上げても落ちないので、**1280x720 MJPG を採用**しました。

UVC カメラは要求を黙って無視することがあるため、`actual_format()` で実際の値を確認してください。

## 制御ループを止めない取得

`cv2.VideoCapture.read()` は次のフレームが来るまでブロックします。制御ループから直接呼ぶと、120 Hz のループが 20 fps に落ちます。

`camera/capture.py` はカメラごとにスレッドを立て、**最新フレームのみを公開**します。制御ループは待たずに現在のフレームを取得します。中間フレームを捨てるのは意図的で、ポリシー推論もテレオペも欲しいのは最新の映像であり、溜まった古い映像ではありません。

### 実測結果

```bash
uv run scripts/bench_camera.py --cameras icspring --port COM4 --seconds 6 \
  --width 1280 --height 720 --fourcc MJPG
```

1280x720 MJPG でストリーミングしながら、サーボを 120 Hz でポーリングした結果です。

| 項目 | 値 |
|---|---|
| サーボループ 中央値 | 1.15 ms |
| サーボループ p95 | 1.40 ms |
| サーボループ 最大 | 2.20 ms |
| 120 Hz の予算 8.3ms 超過 | **0 件（0.0%）** |
| カメラ | 19.1 fps、取得失敗 0 件 |
| 最新フレームの経過時間 | 14 ms |

カメラなしの `bench_latency.py` が中央値 1.00ms だったので、**カメラによる増加は 0.15ms 程度**です。制御ループへの影響はありません。

## 役割の設定

どのカメラが俯瞰でどれが手先かは物理的な配置の問題で、自動検出できません。リポジトリ直下の `cameras.json` に記述します。

```json
{
  "overhead": {
    "name": "icspring",
    "width": 1280,
    "height": 720,
    "fps": 20,
    "fourcc": "MJPG"
  }
}
```

**別のリグではこのファイルを編集してください。** `name` は `find_cameras.py` が表示するデバイス名（の部分一致）です。

LeRobot のロボット設定が期待する形式に変換できます。インデックスは変換時に解決するので、古い番号が設定に焼き付くことはありません。

```python
from so101.camera import to_lerobot
cameras = to_lerobot()   # {"overhead": OpenCVCameraConfig(index_or_path=1, ...)}
```

これをフェーズ 3 の `lerobot-record` にそのまま渡せます。

## 使い方

```python
from so101.camera import CameraSet, load

with CameraSet.from_specs({r: s.name for r, s in load().items()},
                          width=1280, height=720, fourcc="MJPG") as cams:
    cams.wait_for_frames()
    while True:
        frames = cams.read()          # ブロックしない
        image = frames["overhead"].image
```

## 残課題

- [ ] 手先カメラ（グリッパ搭載）の追加。遮蔽への強さが変わります
- [ ] EMEET のカバーを開けて使えるか確認
- [ ] 2 台同時ストリーミング時の USB 帯域確認（同一ハブにサーボ 2 本 + カメラ 2 台）

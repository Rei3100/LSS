"""推論ライブラリを読み込まずに共有するローカルIPCの接続設定。"""

# 3: 分離ジョブが保存先（destination_dir）を指定できる。古い常駐workerへ送らない。
# 4: 単話BGM候補の解析ジョブ（cmd=bgm_analysis）。解析器を持たない古いworkerへ送らない。
# 5: BGM解析ジョブのブロック指定（blocks）と、歌唱回収の4分類ジョブ（cmd=bgm_four_stem）。
#    それらを知らない古いworkerが全体解析や別ジョブとして黙って処理しないよう、接続先を分ける。
# 6: BGM候補の意味判定のためのタグ付けジョブ（cmd=bgm_event_tags）と、SFC解析の継ぎ目の変更。
# 7: 処理中の終了（shutdown の abandon）。実行中の1件を公開の前で止めて保存しない。
#    それを知らない古いworkerが黙って公開しないよう、接続先を分ける。
# 8: 言葉で指定した欲しい音の抽出ジョブ（cmd=wanted_sound）。知らない古いworkerが伴奏の分離として
#    誤って受け付けないよう、接続先を分ける。
VERSION = 8
ADDRESS = rf"\\.\pipe\lss-worker-v{VERSION}"
AUTHKEY = f"lss-worker-v{VERSION}".encode("ascii")
PROBE_TIMEOUT_SECONDS = 1.0

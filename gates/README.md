# 公開前ガードと審査候補の移送

Python 3.11+ と Git だけを使用する。通常の仮想環境は不要。
`LSS_PUBLISH_DENYLIST` selects the private UTF-8 denylist. Otherwise the default
is `publish-denylist.txt` under `LSS_PRIVATE_ROOT`. Missing configuration, missing
files, empty lists and read failures block publication. Blank lines and lines
starting with `#` are ignored. Never display the private list's contents.

```powershell
python -I -B gates/public_guard.py --repo .
python -I -B gates/public_guard.py --repo . --staged
python -I -B gates/public_guard.py --repo . --tree-ish <full-commit-oid>
python -I -B gates/migrate_to_public.py --repo . --source-repo <source-checkout> --source-commit <full-commit-oid>
python -B -m unittest discover -s tests/gates -v
git config --local core.hooksPath .githooks
```

hooks は `python` を使用する。必要なら `LSS_GUARD_PYTHON` に実行可能な
Python の絶対パスを設定する（引数を含めない）。Git for Windows の sh 用。
フックの実行権限を Git index に保存すること。`--staged` は `git write-tree`
で index のスナップショットを取り、unstaged の内容を検査結果に混ぜない。
単独の `--tree-ish` はその commit/tree の内容だけを検査する。
指定コミットのメタデータも秘密形式・禁止語を検査する。
pre-push は stdin の送信 OID を使い、新規の各コミットと tip を検査する。
One ref update is allowed: `refs/heads/main`, or an agent branch under
`refs/heads/codex/` or `refs/heads/claude/` with the same name pattern as
`lss_gh.py`. Every allowed branch receives the full content and ancestor scan.
Deletes, tags and all other refs are blocked. The publication wrapper verifies
the fetch URL and every effective push URL against the fixed repository forms.

`lss_gh.py review-record` reads a JSON object binding `base_sha`,
`review_head_sha`, `diff_digest`, `engine` (`"claude"` or `"codex"`),
`complete: true`, a nonempty `summary`, `findings_p0p1` (integer >= 0) and
`unresolved_p0p1` (integer). Booleans are not integer counts. The input uses
`review_head_sha` and `diff_digest`, not `head_sha` or `bc_diff_digest`.
It validates the current head and posts these same fields in a fenced
`lss-review-record` block accepted by the CI collector and review gate.

The trusted review policy explicitly selects `"bot"` or `"local-records"` mode;
missing or unknown modes block integration. In local-records mode each
implementation engine requires a current completed record from a different
engine: Claude requires Codex, Codex requires Claude, and human requires either.
A mixed Claude/Codex implementation requires both records. Only records with
`unresolved_p0p1 == 0` count; zero findings is a completed review. Freshness binds
the base SHA and B/C digest, with the reviewed head equal to the current head or
an ancestor followed only by class-A changes. The same account can post the
record: this is an omission check, not forgery resistance. Bot mode retains the
bot identity/completion and independent Claude record requirements in `CI.md`.

Deployment retains the seven receipt conditions. Immediately before launching
checkout code, `deploy()` additionally requires HEAD and `origin/main` to equal
the candidate source commit and `git status --porcelain --untracked-files=all`
to be empty. Package roots and children with symlink or Windows reparse-point
attributes, including Python 3.11 junctions, cannot pass the frozen-package check.
ref 削除は内容送信がないため検査対象外。

stdout は JSON 1件、stderr は件数・対象 OID・許可リスト digest の要約。
pre-push では検査したコミットごとの JSON Lines を出力する。
終了値は 0=合格、1=検出、2=走査不成立。0 以外はすべて遮断する。
所在は `path`、`rule`、`line`（パス/構造は0）で示す。
一致本文は常に `<masked>`、秘密形式・禁止語を含むパスは全体をマスクする。
エラーの例外本文や Git の stderr は表示しない。

## 許可リスト

正本は `.publish-allow.json` だけ。初期状態では承認集合を空にしてあるため、
この実装を含む作業ツリー全体もまだ公開承認済みではない。
`include` は Python `fnmatchcase`（`*` は `/` にも一致）の候補パターン。
コードを審査してから正確なパスを `reviewed_code_paths` に登録する。
この集合の既存コードの差分は内容検査を続ける。新規・移動・改名は再登録が必要。
文書・JSON・画像・設定・第三者資産等は個別に `approved_paths` に登録する。

```json
{
  "approved_paths": {
    "docs/example.md": {
      "blob_oid": "<git-hash-object-result>",
      "reason": "Content, rights and privacy reviewed"
    }
  }
}
```

OID は SHA-1 の40桁または SHA-256 の64桁。内容変更で承認が失効する。
`.publish-allow.json` 自身だけ固定パス例外で、自己 OID は不要。
許可リストの本文検査は継続し、その変更も分類Cで審査する。
commit/tree モードでは許可リストも同じツリーから読む。
`--allowlist` は信頼した外部版を明示する場合だけ使用し、結果の digest と一緒に保存する。

## ルール

| rule | 意味 |
|---|---|
| `UNAPPROVED_PATH` / `OID_MISMATCH` | 未審査、または個別承認後の内容変更 |
| `DENIED_PATH` | 優先除外。状態、Evidence、媒体・モデル・venv のディレクトリ、生成物等。個別承認より優先 |
| `INVALID_PATH` / `PATH_COLLISION` | traversal、Windows 特殊パス、大文字小文字の衝突等 |
| `DENYLIST` / `CANARY` | 私的リテラルの大文字小文字を無視した一致、またはカナリア形式 |
| `PRIVATE_KEY` / `TOKEN` / `COOKIE` / `ENV_SECRET` | 秘密鍵、一般的なトークン prefix、Cookie jar/header、長い環境変数/認証値代入 |
| `PRIVATE_PATH` | ドライブ付きユーザーパス、他ドライブの絶対パス、私的保存先、UNC |
| `BINARY_OR_MEDIA` | 拡張子、magic、非 UTF-8、制御バイトによる媒体・重み・archive・実行物・バイナリ判定 |
| `OVERSIZE` | 許可リストの上限超過（初期8 MiB、設定自体の最大100 MiB）。超過本文は読まない |
| `LINK_OR_SUBMODULE` | symlink、Windows reparse point/junction、submodule、特殊ファイル |
| `SCAN_ERROR` / `PUSH_INPUT_ERROR` | 欠落、読取失敗、不正設定、Git失敗、不正な送信入力等 |

個別 OID 承認だけが型検査の例外となる。優先除外・サイズ・秘密・禁止語の例外にはならない。
承認済みバイナリも byte 表現と可能な UTF エンコードを内容検査する。
作業ツリー検査は `.git` 管理領域以外の未追跡・ignore済みファイルも含む。

移送は指定コミットが source HEAD と一致し、コピー対象の index と作業ファイルが
その blob に一致する場合だけ行う。コピーは `git cat-file` で取得した未改変の bytes。
保守的に、CRLF/clean filter 等で作業 bytes が blob と違う対象も拒否する。
destination の `gates/` 外の未コミット変更は拒否し、既存ファイルを上書きしない。
未許可・型不合格・本文不合格等はコピーせず rule ごとの除外件数を manifest に残す。
媒体やモデル等の型不合格は、個別承認があっても移送しない。
`gates/migration-manifest.json` には source commit、各 blob OID、destination path、
SHA-256 と許可リスト digest を記録する。manifest 自身は後で個別審査が必要。
コピー中の I/O 障害では一部ファイルが残り得る。エラーを合格とみなさず、残存物を確認する。

## 保証の範囲

権利・ライセンス・文章の意味・未知の秘密形式・圧縮/暗号化/難読化された内容・
画像やフォントの全メタデータは判断できない。専門 secret scanner の代替認定ではない。
コードを実行して将来作るデータも判断しない。
単独ガードの履歴走査・全 ref 棚卸しは範囲外。初回は別工程で親なし1コミットと
main だけであることを確認する。pre-push の新規コミット検査は履歴混入の補助であり、
リモートの既存履歴を公開承認済みと認定するものではない。
個人メール/ホスト名の一般検出、全第三者通知確認、T7 全体と T13 は別工程。

ローカル hooks は `--no-verify`、hooksPath/コード/許可リスト/env の変更で迂回できる。
同時に別プロセスがファイル・index・refs を書き換える攻撃への隔離/ロックは提供しない。
Git 自体とローカル設定は信頼する。PR/Issue/API/GUI、別 ref・タグの公開、Actions の
ログ/成果物は別の送信制御が必要。CI の検知は送信前の防止にならない。

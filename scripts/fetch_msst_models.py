"""Put the pinned MSST model definitions at vendor/msst/models/, verified file by file. The repository does not carry them.

`vendor/msst/models/` is upstream ZFTurbo/Music-Source-Separation-Training at one commit, byte for byte. It bundles
third-party implementations under their own terms (BandIt Apache-2.0, TorchAudio BSD-2-Clause, Apollo CC-BY-SA-4.0,
a Mamba2 port without a stated licence), so this public tree fetches it from upstream instead of redistributing it.
Each installed file must match its pinned Git blob id; an existing file with another blob id stops the run.

    python scripts/fetch_msst_models.py                 # download the commit archive, verify, install
    python scripts/fetch_msst_models.py --archive <zip> # use an archive already on disk (verified the same way)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "2ba884c2083070c4061fb2d5e3afc41e32420b8a"
ARCHIVE_URL = f"https://github.com/ZFTurbo/Music-Source-Separation-Training/archive/{COMMIT}.zip"
PREFIX = f"Music-Source-Separation-Training-{COMMIT}/models/"
ROOT = Path(__file__).resolve().parents[1]
# path under models/ -> Git blob id at COMMIT
MANIFEST = {
    "bandit/core/__init__.py": "a4d6d7953709c2f86a6b484e49c7715b58bbe86a",
    "bandit/core/data/__init__.py": "1087fe2c4d7d3048295cdf73c0725a015bc0d129",
    "bandit/core/data/_types.py": "9499f9a80b5dec7b5b0e7882849e4f7b2c801ccf",
    "bandit/core/data/augmentation.py": "238214bf17a69e71f48e8761e1ead05b17d0fa5a",
    "bandit/core/data/augmented.py": "84d19599a6579eb5afd304ef6da76a6cbca49045",
    "bandit/core/data/base.py": "a7b6c33a85b93c32209138e3d21bfc8e0f270cac",
    "bandit/core/data/dnr/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "bandit/core/data/dnr/datamodule.py": "dc5550608aabf460eb1781576112ed60185dd318",
    "bandit/core/data/dnr/dataset.py": "7b241cf0dd474eafbfc9db3ec2f4987d12596de4",
    "bandit/core/data/dnr/preprocess.py": "9d0b58690f3bae726b0655dbade6393c89bf8c9e",
    "bandit/core/data/musdb/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "bandit/core/data/musdb/datamodule.py": "a8984daebd535b25f0551d348c91dbd1702fb9da",
    "bandit/core/data/musdb/dataset.py": "53f5d9afdfe383600b5f89767c4ef1f4b54f4a47",
    "bandit/core/data/musdb/preprocess.py": "5d5892e5c3f4acef5bbb183a746d76475c461810",
    "bandit/core/data/musdb/validation.yaml": "2f8752478d285d1d13d5e842225af1de95cae57a",
    "bandit/core/loss/__init__.py": "0ab803aecde4f686e34d93f3f2d585e0a9867525",
    "bandit/core/loss/_complex.py": "1d97e5d8bab3fdb095c2ba7c77faebef26e8f1ce",
    "bandit/core/loss/_multistem.py": "675e0ffbecf1f9f5efb0369bcb9d5c590efcfc31",
    "bandit/core/loss/_timefreq.py": "6ea9d5994ca645546b5ccb7e6eafaa3d2fbcf959",
    "bandit/core/loss/snr.py": "2996dd57080db687599c1fd673d6807041a04b52",
    "bandit/core/metrics/__init__.py": "c638b4df585ad6c3c6490d9e67b7fc197f0d06f4",
    "bandit/core/metrics/_squim.py": "ec76b5fb5e27d0f6a6aaa5ececc5161482150bfc",
    "bandit/core/metrics/snr.py": "d2830b2cbecfa681c449d09e2d4c35a20fc98128",
    "bandit/core/model/__init__.py": "54ac48eb69d6f844ba5b73b213eae4cfab157cac",
    "bandit/core/model/_spectral.py": "564cd28600719579227a6085eed5e9d6ee521312",
    "bandit/core/model/bsrnn/__init__.py": "c27826197fc8f4eb7a7036d8037966a58d8b38d4",
    "bandit/core/model/bsrnn/bandsplit.py": "63e6255857fb2d538634be317332afb2f93e145d",
    "bandit/core/model/bsrnn/core.py": "7fd36259002a395e7b7864f605fcab5b4422e422",
    "bandit/core/model/bsrnn/maskestim.py": "0b9289dfa702e02ff4d4f0dc76196fd39bb68e34",
    "bandit/core/model/bsrnn/tfmodel.py": "ba710798c5ab49936bd63c914f20da516cbc6af9",
    "bandit/core/model/bsrnn/utils.py": "bf8636e65fe9e7fdd13fa063760018df90a01cff",
    "bandit/core/model/bsrnn/wrapper.py": "a31c087db33eb215effa3c3fc492999c5672c55e",
    "bandit/core/utils/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "bandit/core/utils/audio.py": "adae756bf2b02a994a42fcc007da1e1ff7bb6cfb",
    "bandit/model_from_config.py": "00ea586d7dfdbd6b89d6b7f2f400e6c8d04da5e4",
    "bandit_v2/bandit.py": "ac4e13f479891065cf1f7dae0720721128347979",
    "bandit_v2/bandsplit.py": "a14ea52bfa318264d536c9f934d0e28db63e15dc",
    "bandit_v2/film.py": "e30795332ea0e06865ea3d883767db17bb02353c",
    "bandit_v2/maskestim.py": "65215d86a5e94dafdb71744aafadf7aaab93330d",
    "bandit_v2/tfmodel.py": "21aef03d1f0e814c20db05fe7d14f8019f07713b",
    "bandit_v2/utils.py": "ad4eab5d8c5b5396ed717f5b9c365a6900eddd2f",
    "bs_mamba2_code/attend_mamba.py": "9256c18580fd5787cd9e3deb19afd936dad8e0ed",
    "bs_mamba2_code/bs_mamba2.py": "5cde6d14c819f45fad2c82ca68372f28df1d5b35",
    "bs_roformer/__init__.py": "d6923b6367ba6b98c65118f8728ac427d00847a2",
    "bs_roformer/attend.py": "d6dc4b3079cff5b3c8c90cea8df2301afd18918b",
    "bs_roformer/bs_conformer.py": "ae370dc97cfaf5bab7f83c25c1c905b1e887a52a",
    "bs_roformer/bs_roformer.py": "60fa27942e09fbdf4f9463259fd0b80582d9e39c",
    "bs_roformer/bs_roformer_experimental.py": "a5c3544374d16c9429272f69b4621bc3878093cc",
    "bs_roformer/mel_band_conformer.py": "dda19a1a7ba4a29128873587568d6092faa18ca0",
    "bs_roformer/mel_band_roformer.py": "b49235bbdd99c2fcabc5fefe747322f37ef84454",
    "bs_roformer/mel_band_roformer_experimental.py": "5e9afeda0cd9db0a95cf3fed67034ddc4695d9f2",
    "conformer_model.py": "d5f980deca90034a26070f2212a91c61bb14409c",
    "demucs4ht.py": "06c279c31a7ac7e12af4375a5715eb291ad5405c",
    "ex_bi_mamba2.py": "6bcf797cce8b28e88c7aab9adfa2037070504763",
    "look2hear/models/__init__.py": "b72fb1f70bbc43f10bd74583d582ddb45848e45e",
    "look2hear/models/apollo.py": "5de9afd468e4635e581c0ff41dce7acc4eb249be",
    "look2hear/models/base_model.py": "e24b23192141634c0496be04899b6ecdb683b6c5",
    "mdx23c_tfc_tdf_v3.py": "95a555b27c03e0e919ad2015d350c84b9db2f7e6",
    "mdx23c_tfc_tdf_v3_with_STHT.py": "2472fed312fa8c07dd33fa8510249bac941cb9d4",
    "mel_band_conformer.py": "31e066ada2b9ac83ef2b5bead05be5ce5388b373",
    "scnet/__init__.py": "5f6ecefede9345237623066dd21ebd8253af1c60",
    "scnet/scnet.py": "b27704dc922eb593dc76f3b9905aa8c0ea02507f",
    "scnet/scnet_masked.py": "c4354e86235261d0459a51ee74384aedcaa7903c",
    "scnet/scnet_tran.py": "3f3bd05b3c1504ab807ff2103a81b4a5bf536700",
    "scnet/separation.py": "d902dac4d947123d3ba1270dd065be0d8b4c5ed9",
    "scnet_unofficial/__init__.py": "6d034d38a2ace2e81bd28d63dd8f25feb918f33d",
    "scnet_unofficial/modules/__init__.py": "69617bb15044d9bbfd0211fcdfa0fa605b01c048",
    "scnet_unofficial/modules/dualpath_rnn.py": "2dfcdbcfc102a6fde5a2ff53a2a06f2d6caae196",
    "scnet_unofficial/modules/sd_encoder.py": "742577f480693671437dc50358a1a65d251b6e9b",
    "scnet_unofficial/modules/su_decoder.py": "660c1fa6cbfd9b43bed73204a0bb6593524de272",
    "scnet_unofficial/scnet.py": "d076f85f1d5ce1345dc9a8c56b6a5aef09f2facc",
    "scnet_unofficial/utils.py": "aae1afcd52e8088926ea984e52c9b62ca68be65c",
    "segm_models.py": "5e8f1814a30bd59d4cd94bb4a19edf8d409c24b5",
    "torchseg_models.py": "419abd3bc0b49424e9ba2f6f4f604d2e4f2938ea",
    "ts_bs_mamba2.py": "a01a2a58c5418bfd846d6edb6900810392091189",
    "upernet_swin_transformers.py": "ddf61cc3785877c7e65d82bf81435a1a3fff0f1b",
}


def tls_context():
    """System trust plus certifi's roots when available (a venv's OpenSSL may lack the issuer)."""
    import ssl

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class FetchError(RuntimeError):
    pass


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def install(archive: Path, target: Path, manifest: dict[str, str] = MANIFEST, prefix: str = PREFIX) -> dict:
    """Verify every manifest member of `archive` before writing any of them; return counts."""
    with zipfile.ZipFile(archive) as bundle:
        members = {}
        for name, oid in manifest.items():
            try:
                data = bundle.read(prefix + name)
            except KeyError as error:
                raise FetchError(f"archive lacks {name}") from error
            if blob_id(data) != oid:
                raise FetchError(f"{name} does not match the pinned blob")
            members[name] = data
    counts = {"installed": 0, "present": 0}
    for name, data in members.items():
        path = target / name
        if path.exists():
            if blob_id(path.read_bytes()) != manifest[name]:
                raise FetchError(f"{name} exists with other content; move it away to install the pinned files")
            counts["present"] += 1
    for name, data in members.items():
        path = target / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(data)
        os.replace(partial, path)
        counts["installed"] += 1
    return counts


def complete(target: Path, manifest: dict[str, str] = MANIFEST) -> bool:
    return all((target / name).is_file() and blob_id((target / name).read_bytes()) == oid
               for name, oid in manifest.items())


def ensure(target: Path, archive: Path | None = None, url: str = ARCHIVE_URL, **pins) -> dict:
    if complete(target, pins.get("manifest", MANIFEST)):
        return {"installed": 0, "present": len(pins.get("manifest", MANIFEST))}
    if archive is not None:
        return install(archive, target, **pins)
    with tempfile.TemporaryDirectory(prefix="lss-msst-") as scratch:
        downloaded = Path(scratch) / "msst.zip"
        with urllib.request.urlopen(url, timeout=60, context=tls_context()) as response, downloaded.open("wb") as sink:
            for block in iter(lambda: response.read(1 << 20), b""):
                sink.write(block)
        return install(downloaded, target, **pins)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, help="commit archive already downloaded")
    parser.add_argument("--target", type=Path, default=ROOT / "vendor" / "msst" / "models")
    args = parser.parse_args(argv)
    try:
        counts = ensure(args.target, args.archive)
    except (FetchError, OSError, zipfile.BadZipFile) as error:
        print(f"fetch_msst_models: FAILED: {error}", file=sys.stderr)
        return 1
    print(f"fetch_msst_models: {counts['installed']} installed, {counts['present']} present at {COMMIT[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

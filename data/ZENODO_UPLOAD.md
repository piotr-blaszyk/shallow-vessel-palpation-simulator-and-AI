# Uploading the data bundle to Zenodo from the command line

**Short answer: yes.** A `.tar.gz` can be uploaded entirely from an Ubuntu
terminal — the web form is not required. Zenodo's REST API accepts a plain
`curl` PUT, which is also *more reliable* than the browser for large files,
because it can be retried and resumed rather than restarting from zero.

Everything below needs only `curl` and `jq` (`sudo apt install curl jq`).

## 1. Create an access token (one-time, in the browser)

Zenodo → *Account* → *Applications* → *Personal access tokens* → *New token*.
Tick the scopes **`deposit:write`** and **`deposit:actions`**.

Keep it out of your shell history and out of git:

```bash
read -rs ZENODO_TOKEN && export ZENODO_TOKEN
```

## 2. Create a draft deposition

```bash
curl -s -H "Authorization: Bearer $ZENODO_TOKEN" \
     -H "Content-Type: application/json" \
     -X POST https://zenodo.org/api/deposit/depositions \
     -d '{}' | tee deposition.json | jq '{id, links}'
```

Capture the two values the next steps need:

```bash
DEPOSITION_ID=$(jq -r .id deposition.json)
BUCKET_URL=$(jq -r .links.bucket deposition.json)
```

## 3. Upload the tarball

The bucket API streams the file, so a multi-GB archive is fine. `--retry`
makes a dropped connection recoverable, which is the main advantage over the
web uploader:

```bash
curl --progress-bar \
     --retry 5 --retry-delay 5 --retry-connrefused \
     -H "Authorization: Bearer $ZENODO_TOKEN" \
     --upload-file shallow-vessel-palpation-data.tar.gz \
     "$BUCKET_URL/shallow-vessel-palpation-data.tar.gz" | jq '{key, size, checksum}'
```

Verify the reported `checksum` (an MD5) against the local file:

```bash
md5sum shallow-vessel-palpation-data.tar.gz
```

## 4. Attach metadata

```bash
curl -s -H "Authorization: Bearer $ZENODO_TOKEN" \
     -H "Content-Type: application/json" \
     -X PUT "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID" \
     -d @- <<'JSON' | jq '.metadata.title'
{
  "metadata": {
    "title": "Data for: Sim-to-Real Subsurface Feature Localisation with an Optical Tactile Sensor",
    "upload_type": "dataset",
    "description": "Simulated training dataset, preprocessed real silicone- and meat-phantom tactile trials, and trained GNN checkpoints accompanying the shallow-vessel-palpation-simulator-and-AI and shallow-vessel-palpation-robot-control repositories. See MANIFEST.md inside the archive for a full description of contents.",
    "creators": [
      {"name": "Blaszyk, Piotr"}
    ],
    "access_right": "open",
    "license": "cc-by-4.0"
  }
}
JSON
```

## 5. Publish

Publishing mints the DOI and is **irreversible** — files cannot be removed from
a published record afterwards (only superseded by a new version). Check the
draft in the browser first, then:

```bash
curl -s -H "Authorization: Bearer $ZENODO_TOKEN" \
     -X POST "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/actions/publish" \
     | jq '{doi, links: .links.record_html}'
```

## Sandbox first (recommended)

Rehearse the whole flow against `https://sandbox.zenodo.org` (separate account
and separate token) by substituting the host in every URL above. Sandbox records
are disposable, so a mistake costs nothing.

## One file or two?

One archive is simpler to cite and to download. Split into two only if you also
want to publish the **raw** experimental recordings (~1.6 GB of `.avi`), which
are excluded from the default bundle — in that case ship
`shallow-vessel-palpation-data.tar.gz` (the ~186 MB reproduction bundle) and
`difftactile-raw-recordings.tar.gz` separately, so users who only want to verify
results are not forced to download the raw videos.

## Alternative: a wrapper tool

If you would rather not hand-roll `curl`, `zenodo-uploader` and `zenodo_client`
(both `pip install`-able) wrap the same API. They add nothing the commands above
lack, so the plain-`curl` route is preferred here — no extra dependency, and
every step is visible.

Sources:
- [Zenodo REST API — developers.zenodo.org](https://developers.zenodo.org/)
- [jhpoelen/zenodo-upload (curl + bash reference implementation)](https://github.com/jhpoelen/zenodo-upload)
- [zenodo-uploader on PyPI](https://pypi.org/project/zenodo-uploader/)
- [zenodo_client on PyPI](https://pypi.org/project/zenodo-client/)

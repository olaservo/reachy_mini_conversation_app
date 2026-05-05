## Test procedure

### Prerequisites

```bash
pip install huggingface-hub
```

### 1. Download the HF space snapshot (creates symlinked cache)

```bash
SNAPSHOT=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('pollen-robotics/reachy_mini_conversation_app', repo_type='space'))
")
echo "$SNAPSHOT"
```

### 2. Confirm the build fails on `main`

```bash
cd "$SNAPSHOT"
rm -rf build src/reachy_mini_conversation_app.egg-info
python3 -m pip wheel --no-deps --no-build-isolation .
```

Expected error:
```
error: [Errno 2] No such file or directory: '…/blobs/profiles'
```

### 3. Apply the fix

```bash
# setup.py is a symlink → blob; patch the blob in-place
BLOB="$(cd "$SNAPSHOT" && readlink setup.py)"
sed -i.bak 's/Path(__file__).resolve().parent/Path(__file__).parent.resolve()/' "$SNAPSHOT/$BLOB"
```

### 4. Confirm the build succeeds

```bash
cd "$SNAPSHOT"
rm -rf build src/reachy_mini_conversation_app.egg-info
python3 -m pip wheel --no-deps --no-build-isolation .
```

Expected: `Successfully built reachy_mini_conversation_app`

### 5. Restore the cache

```bash
mv "$SNAPSHOT/$BLOB.bak" "$SNAPSHOT/$BLOB"
```

### Why this happens

`snapshot_download` stores every file as a symlink into a `blobs/` directory. `Path(__file__).resolve().parent` follows the symlink first, so `PROJECT_ROOT` lands in `blobs/` instead of the snapshot directory where `profiles/` lives. Swapping to `.parent.resolve()` gets the parent before resolving the symlink.

# pyDEXPI Pump-Pipe-Tank POC

Tiny proof-of-concept for loading and rendering a minimal DEXPI XML:

Pump (E1) -> Pipe -> Tank (E2)

## Files

- `sample.xml`: Minimal DEXPI sample model.
- `poc.py`: Loads XML with `ProteusSerializer` and tries rendering with `SvgRenderer`.
- `requirements.txt`: Python dependency list.

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the POC:

   ```bash
   python test.py
   ```

## Expected behavior

- The script loads `sample.xml` and prints model/debug info.
- If rendering succeeds, it writes `output.svg`.
- If rendering fails, the script prints diagnostics and a likely cause.

## Notes

This sample is intentionally minimal. Some `pyDEXPI` workflows may expect fuller Proteus-exported XML (for example geometry, ports, and related details).

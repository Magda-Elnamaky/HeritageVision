# HeritageVision & Match — Worldwide Reference UI

This build uses the supplied reference artwork as the Home hero and rebuilds the surrounding page as interactive Streamlit components. The sidebar navigation, archive, map, image search, maker section, and world-region cards are real controls. The hero image itself is artwork, so its printed words/buttons are decorative; functional controls are placed around it.

## Run

1. Open PowerShell in this folder.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the app: `python -m streamlit run app.py`

## Local image index (optional)

The app's local visual similarity search requires images under `data/images/<country-or-collection>/`. Add your own licensed images first, then run `python build_index.py`. The first run may download pretrained ResNet50 weights and requires internet access. You can use the global online search pages without creating a local image index.

The map starts with a curated set of real coordinates and official/source links; it is not an exhaustive global heritage gazetteer.

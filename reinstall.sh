UPDATE_IMPORTS=${1:-0}
if [ $UPDATE_IMPORTS -eq 1 ]; then
    echo "Updating imports"
    python update_imports.py
fi
pip install --force-reinstall --no-deps -e .

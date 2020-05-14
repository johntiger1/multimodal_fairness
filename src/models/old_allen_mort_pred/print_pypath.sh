# in theory, this command should work:
# allennlp train experiments/venue_classifier.json -s tmp/venue/out_dir --include-package text_mortality
# the reason is that `import text_mortality` works

python -c "import sys; print(sys.path)"

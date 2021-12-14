- [ ] Update version number in `psiaudio/__init__.py`
- [ ] Commit the changes
- [ ] Build sdist and wheel `python setup.py sdist bdist_wheel --universal`
- [ ] Upload to test server via twine `twine upload -r testpypi dist/*`
- [ ] Create test env: `conda create -n test python` 
- [ ] Install in test env: 
```
conda activate test
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple psiaudio
python -m pip install pytest pytest-benchmark
cd <src folder>/tests
pytest
```
- [ ] Verify downloaded version was used: `python -c "import psiaudio; print(psiaudio.__version__)"`
- [ ] Verify downloaded package was used: `python -c "import psiaudio; print(psiaudio.__file__)"`
- [ ] Upload to actual server via twine `twine upload dist/*`

- [ ] Commit the changes
- [ ] Tag with the version number, e.g., `git tag -a -m "Release 0.2.0" 0.2.0`.
- [ ] Verify setuptools_scm picks up the version number using `python -m setuptools_scm`.
- [ ] Build sdist and wheel using `python build .`.
- [ ] Upload to test server via twine `twine upload -r testpypi dist/*`.
- [ ] Create test env: `conda create -n test python`.
- [ ] Install in test env: 
```
conda activate test
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple psiaudio pytest
cd <src folder>/tests
pytest
```
- [ ] Verify downloaded version was used: `python -c "import psiaudio; print(psiaudio.version.version)"`.
- [ ] Verify downloaded package was used: `python -c "import psiaudio; print(psiaudio.__file__)"`.
- [ ] Upload to actual server via twine `twine upload dist/*`.

Releases are built and published by GitHub workflows; there is nothing to
build or upload by hand.

- [ ] Commit the changes and merge them into `main`.
- [ ] Tag `main` with the version number, e.g.,
  `git tag -a -m "Release 0.2.0" 0.2.0`. setuptools_scm takes the version
  from the tag.
- [ ] Push `main` and the tag: `git push origin main 0.2.0`.
  - Pushing a tag of the form `X.Y.Z` runs `publish-to-pypi.yml`, which
    builds the sdist and wheel and publishes them to PyPI.
  - Pushing `main` runs `deploy-docs.yml`, which rebuilds the documentation.
- [ ] Check that both runs succeeded (`gh run list`) and that the new version
  is on PyPI (`python -m pip index versions psiaudio`).
- [ ] Only then push packages that require the new version (e.g.,
  psiexperiment). Their test and docs workflows install psiaudio from PyPI
  and fail if it has not been published yet. If they ran too early, re-run
  them (`gh run rerun <run id>`).

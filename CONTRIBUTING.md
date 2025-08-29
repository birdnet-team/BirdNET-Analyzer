### Testdata (optional)

This project uses a submodule for its testdata.

If you want to run the tests locally you need to initialize the submodule:

```bash
git submodule update --init --recursive
```

Or checkout the submodule during cloning:

```bash
git clone --recurse-submodules https://github.com/birdnet-team/BirdNET-Analyzer.git
```

If the testdata has changed and you want to update the repo to use the new current testdata follow these steps:

```bash
cd tests/data
git pull origin main
cd ..
git add data
git commit -m "Update testdata"
git push
```

This will set up the repo to use the current ref of the submodule.

**Note**: If you wan to add data you do so by either checking out the submodule as a regular repo and just add-commit-push from there or use the subdirectory containing the submodule like a regular git repository (don't forget to manually update the ref in the main repo)
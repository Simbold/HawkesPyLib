rm -rf docs/*
pdoc --html --template-dir=pdoc/templates --force -o pdoc/ HawkesPyLib
mv pdoc/HawkesPyLib/* docs/
rm -rf pdoc/HawkesPyLib
cp pdoc/logo.png docs

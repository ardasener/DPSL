METIS_LINK="http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
PULP_LINK="https://github.com/HPCGraphAnalysis/PuLP.git"

# Clean & Create directories
rm -rf ./temp
rm -rf ./libs
mkdir -p temp
mkdir -p libs

# METIS
cd temp
wget -c $METIS_LINK
tar xzf "metis-5.1.0.tar.gz"
cd "metis-5.1.0"
make config
make
cd ../..
mv ./temp/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.a libs/
mv ./temp/metis-5.1.0/include/metis.h libs/

# PULP
cd temp
git clone $PULP_LINK pulp
cd pulp
./install
cd ../..
mv ./temp/pulp/lib/* libs/
mv ./temp/pulp/include/* libs/

# Clean
rm ./temp -rf
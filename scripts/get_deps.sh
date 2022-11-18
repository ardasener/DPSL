MAKE_J=30

MTMETIS_VERSION=0.7.2
MTMETIS_LINK="http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/mt-metis-${MTMETIS_VERSION}.tar.gz"

METIS_VERSION=5.1.0
METIS_LINK="http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-${METIS_VERSION}.tar.gz"

PULP_LINK="https://github.com/HPCGraphAnalysis/PuLP.git"

MTKAHYPAR_COMMIT="f21a4195f66f1a63b252482cdf819ae0e8acdcd0"
MTKAHYPAR_LINK="https://github.com/kahypar/mt-kahypar.git"

# Clean & Create directories
rm -rf ./deps
rm -rf ./libs
mkdir -p deps
mkdir -p libs

# MT-Kahypar
if [ "$1" == "--mtkahypar" ]; then
    echo "Building mtkahypar... (This step requires Boost!)"
    cd deps
    git clone --recursive $MTKAHYPAR_LINK
    cd mt-kahypar
    git checkout $MTKAHYPAR_COMMIT
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_USE_COMPATIBLE_TBB_VERSION=ON
    echo "Next step might give an error, ignore it"
    make mtkahypargq mtkahypargraph -j $MAKE_J
    cd ../../..
    mv ./deps/mt-kahypar/build/lib/*.so libs/
    mv ./deps/mt-kahypar/include/*.h libs/
    mv ./deps/mt-kahypar/build/external_tools/tbb/tbb/lib/intel64/gcc4.8/*.so libs/
else
    echo "Skipping mtkahypar"
fi

# METIS
echo "Building metis..."
cd deps
wget -c $METIS_LINK
tar xzf "metis-${METIS_VERSION}.tar.gz"
cd "metis-${METIS_VERSION}"
make config shared=1
make -j $MAKE_J
cd ../..
mv ./deps/metis-${METIS_VERSION}/build/Linux-x86_64/libmetis/*.so libs/
mv ./deps/metis-${METIS_VERSION}/include/*.h libs/

# MTMETIS
echo "Building mtmetis..."
cd deps
wget -c $MTMETIS_LINK
tar xzf "mt-metis-${MTMETIS_VERSION}.tar.gz"
cd "mt-metis-${MTMETIS_VERSION}"
./configure --shared
make -j $MAKE_J
cd ../..
mv ./deps/mt-metis-${MTMETIS_VERSION}/build/Linux-x86_64/lib/*.so libs/
mv ./deps/mt-metis-${MTMETIS_VERSION}/include/*.h libs/

# PULP
echo "Building pulp..."
cd deps
git clone $PULP_LINK pulp
cd pulp
./install
cd ../..
mv ./deps/pulp/lib/* libs/
mv ./deps/pulp/include/* libs/

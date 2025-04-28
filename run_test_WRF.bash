#!/bin/bash

PID=${BASHPID}

#--------------------------------------------------------------------------
#
#  USER INPUT
#
#-------------------------------------------------------------------------

USE_DEFAULT_NAMELIST=1
# default start_date, end_date, and other parameters are set below
# look for "DEFAULT NAMELIST".

RUN_COMPILE_WRF=0
RUN_COMPILE_WPS=0
FORCE_CLEAN_COMPILE=0

RUN_WPS=0
RUN_REAL_NOCHEM=0
RUN_MEGAN=0
RUN_WESELY_EXO_COLDENS=0
RUN_REAL_CHEM=0
RUN_MOZBC=0
RUN_WRFCHEMI=0
RUN_WRF=1

WRF_CHEM=1
WRF_KPP=1
if [[ $WRF_CHEM -eq 0 ]]; then
  RUN_REAL_CHEM=0
  RUN_WRFCHEMI=0
fi

module_intel="intel/19.0.8.324"
module_openmpi="openmpi/4.0.7"
module_netcdfc="netcdf-c/4.7.4"
module_netcdff="netcdf-fortran/4.5.3"
module_hdf5="hdf5/1.10.7"
module_jasper="jasper/2.0.32"

#--- set NETCDF_DIR.
#    You can use /home/onishi/NETCDF_DIR2/
#    NETCDF_DIR="/home/onishi/NETCDF_DIR2/"
#    OR 
#    set your own directory defined in "install_tools.bash"
NETCDF_DIR="/path/to/merged/NETCDF/directory"


#------------------------------------------------------------------------------

LAUNCH_DIR=$PWD
echo "run_test_WRF.bash is launched in $LAUNCH_DIR"

#---- WRF_DIR         : Absolute path to the top directory of your WRF/WRF-Chem codes to test
#     e.g.) WRF_DIR="/proju/wrf-chem/${USER}/WRF4/WRF_LATMOS/"
WRF_DIR="/path/to/WRF/directory/"
echo "wrf.exe in $WRF_DIR"

#---- WPS_DIR         : Absolute path to the top directory of WPS codes
#     e.g.) WPS_DIR="/home/${USER}/WPS/"
WPS_DIR="/path/to/WPS/directory/"
echo "wps exe files in $WPS_DIR"

#---- WPSdomain_DIR   : Absolute path to a directory where you run wps (ungrib.exe,geogrid.exe,metgrid.exe)
#     e.g.) WPSdomain_DIR="/proju/wrf-chem/${USER}/WPS_domains/TEST2"
WPSdomain_DIR="/path/to/your/WPS/domain/where/you/run/wps/"
echo "WPS domain directory : $WPSdomain_DIR"

#---- WRFrun_DIR      : Absolute path to a directory where you run wrf.exe (working directory)
#     e.g.) WRFrun_DIR="/proju/wrf-chem/${USER}/WRFruns/WRFrun_Test/"
WRFrun_DIR="/path/to/directory/where/you/run/wrf.exe/or/a/batch/script/"
echo "WRF run directory : $WRFrun_DIR"

#---- MEGAN_DIR       : Absolute path to a directory where you run your executable "megan_bio_emiss"
#     e.g.) MEGAN_DIR="/proju/wrf-chem/${USER}/MEGAN/WRFrun_TEST/"
MEGAN_DIR="/path/to/directory/where/you/run/your/executable/megan_bio_emiss/"

#---- MEGAN_exe       : Absolute path to your megan executable "megan_bio_emiss"
#     e.g.) MEGAN_exe="/proju/wrf-chem/${USER}/MEGAN/MEGAN/megan_bio_emiss"
MEGAN_exe="/path/to/your/executable/megan_bio_emiss"

#---- MEGAN_data_DIR  : Absolute path to a directory where you keep input data files for MEGAN
#     e.g.) MEGAN_data_DIR="/proju/wrf-chem/${USER}/MEGAN/DATA_TOTAL"
MEGAN_data_DIR="/path/to/directory/where/you/keep/MEGAN/input/files/"

#---- WESCOL_DIR      : Absolute path to a directory where you run "wesely" and "exo_coldens"
#     e.g.) WESCOL_DIR="/proju/wrf-chem/${USER}/wes-coldens/"
WESCOL_DIR="/path/to/directory/where/you/run/wesely/and/exo_coldens/"

#---- wesely_exe      : Absolute path to your wesely executable "wesely"
#     e.g.) wesely_exe="/proju/wrf-chem/${USER}/wes-coldens/wesely"
wesely_exe="/path/to/your/executable/wesely"

#---- exo_coldens_exe : Absolute path to your exo_coldens executable "exo_coldens"
#     e.g.) exo_coldens_exe="/proju/wrf-chem/${USER}/wes-coldens/exo_coldens"
exo_coldens_exe="/path/to/your/executable/exo_coldens"

#---- MOZBC_DIR       : Absolute path to a directory where you run your executable "mozbc"
#     e.g.) MOZBC_DIR="/proju/wrf-chem/${USER}/MOZBC/MOZBC_TEST/"
MOZBC_DIR="/path/to/directory/where/you/run/mozbc/"

#---- mozbc_exe       : Absolute path to your mozbc executable "mozbc"
#     e.g.) mozbc_exe="/proju/wrf-chem/${USER}/MOZBC/mozbc"
mozbc_exe="/path/to/your/executable/mozbc"

#---- mozbc_data      : Absolute path to input data files for "mozbc"
#     You need to download, for ex. from https://www.acom.ucar.edu/cesm/subset.shtml
#     e.g.) mozbc_data="/proju/wrf-chem/${USER}/MOZART/cesm-ALPACA-0001.nc"
#       
mozbc_data="/path/to/MOZBC/input/files/for/example/cesm-ALPACA-0001.nc"

#---- wrfchemi_python : Absolute path to a python script to create wrfchemi files (1)
wrfchemi_python="${LAUNCH_DIR}/create_wrfchemi.py3"
#---- wrfchemi_DIR    : Absolute path to a directory where wrfchemi files will be created.
wrfchemi_DIR="${LAUNCH_DIR}/WRFrun_TEST/"


#--- Default start and end date and time ----------------------
#    Unless you want to use your own. 
#    keyword : DEFAULT NAMELIST

STARTDATETIME="2022/02/02 00:00:00"
ENDDATETIME="2022/02/04 00:00:00"

USE_POLAR=1
USE_LAMBERT=0

if [[ $USE_POLAR -eq 1 ]]; then
  xNN=50
  yNN=50
  DDx=20000
  RefLat=65
  RefLon=-148
  TrueLat1=${RefLat}
  TrueLat2=90
  StandLon=${RefLon}
  map_proj="polar"
fi

if [[ $USE_LAMBERT -eq 1 ]]; then
  xNN=50
  yNN=51
  DDx=50000
  RefLat=35
  RefLon=0
  TrueLat1=${RefLat}
  TrueLat2=${RefLat}
  StandLon=${RefLon}
  map_proj="lambert"
fi

echo "cell numbers in west-east : $xNN"
echo "cell numbers in south-north : $yNN"
echo "cell resolution : $(($DDx/1000)) km"  
echo "ref_lat : $RefLat"
echo "ref_lon : $RefLon"

TIMESTEP=$(($DDx/1000 * 3))
echo "TIMESTEP : $TIMESTEP sec"

CHEMOPT=202
echo "chem_opt for a WRF-Chem run : $CHEMOPT" 

#--- RUN COMPILE WRF ------------------------------------------------------------------
if [[ $RUN_COMPILE_WRF -eq 1 ]]; then

echo
echo '-------------------------------------------------------------------------'
echo
echo '         COMPILE WRF        '
echo
echo '-------------------------------------------------------------------------'
echo

cd $WRF_DIR
cat > install_wrf.slurm <<EOF2
#!/bin/bash
###SBATCH --account=wlm@cpu
#SBATCH --job-name=CompileWRFChem           # nom du job
#SBATCH --partition=zen16                   # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
###SBATCH --cpus-per-task=8                   # 5 x 4GB = 20 GB memory
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
###SBATCH --mem=40GB                          # memory limit
#####SBATCH --hint=nomultithread                # 1 pocessur MPI par coeur physique
#SBATCH --time=04:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CompileWRFChem%j.out       # nom du fichier de sortie
#SBATCH --error=CompileWRFChem%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

# Link the necessary files from the SI directory

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
echo $PWD

export WRFDIR=\$PWD
module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}
#export NETCDFPAR=/home/onishi/NETCDF_DIR2/
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}
export MPI_LIB=-L\${OPENMPI_ROOT}/lib


export JASPERLIB=\$JASPER_ROOT/lib
export JASPERINC=\$JASPER_ROOT/include
export HDF5_PATH=\$HDF5
## default setting ##
export EM_CORE=1
export NMM_CORE=0
#export WRF_EM_CORE=1
#export WRF_NMM_CORE=0
## end of default setting ##
export WRF_CHEM=$WRF_CHEM
export WRF_KPP=$WRF_KPP
export YACC="/usr/bin/byacc -d"
export FLEX_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export HDF5_DISABLE_VERSION_CHECK=1
export WRFIO_NCD_NO_LARGE_FILE_SUPPORT=0
export WRFIO_NCD_LARGE_FILE_SUPPORT=1


cd \$WRFDIR

latest_exe=\$(find ./ -type f -executable -printf "%T@ %p\\n" | sort -nr | head -1 | awk '{print \$1}')

echo "FORCE_CLEAN_COMPILE=$FORCE_CLEAN_COMPILE"
if [[ "$FORCE_CLEAN_COMPILE" -eq 1 ]] || find ./Registry -type f -newermt "@\$latest_exe" | grep -q .; then
if [[ "$FORCE_CLEAN_COMPILE" -eq 0 ]]; then
  echo "FORCE_CLEAN_COMPILE is zero, but apparently Registry has been modified. "
fi
echo "Running ./clean -aa and ./clean -a"
./clean -aa
./clean -a

cd \$WRFDIR/chem/KPP/kpp/kpp-2.1/src/
/usr/bin/flex scan.l

sed -i '
1 i \\
#define INITIAL 0 \\
#define CMD_STATE 1 \\
#define INC_STATE 2 \\
#define MOD_STATE 3 \\
#define INT_STATE 4 \\
#define PRM_STATE 5 \\
#define DSP_STATE 6 \\
#define SSP_STATE 7 \\
#define INI_STATE 8 \\
#define EQN_STATE 9 \\
#define EQNTAG_STATE 10 \\
#define RATE_STATE 11 \\
#define LMP_STATE 12 \\
#define CR_IGNORE 13 \\
#define SC_IGNORE 14 \\
#define ATM_STATE 15 \\
#define LKT_STATE 16 \\
#define INL_STATE 17 \\
#define MNI_STATE 18 \\
#define TPT_STATE 19 \\
#define USE_STATE 20 \\
#define COMMENT 21 \\
#define COMMENT2 22 \\
#define EQN_ID 23 \\
#define INL_CODE 24
' \$WRFDIR/chem/KPP/kpp/kpp-2.1/src/lex.yy.c

cd \$WRFDIR

sed -i '/^[[:blank:]]*image.inmem_=1/c\\/\*      image.inmem_=1;   \*\/' \$WRFDIR/external/io_grib2/g2lib/enc_jpeg2000.c

sed -i '/I_really_want_to_output_grib2_from_WRF/s/FALSE/TRUE/g' \$WRFDIR/arch/Config.pl

### 13. (serial)  14. (smpar)  15. (dmpar)  16. (dm+sm)   INTEL (ifort/icc)
###./configure -d <<EOF
###./configure -D <<EOF
./configure <<EOF
15

EOF

sed -ie 's/hdf5hl_fortran/hdf5_hl_fortran/' ./configure.wrf

sed -ie '/^CPP /s/$/ -I\$(WRF_SRC_ROOT_DIR)\/external\/ioapi_share/' ./configure.wrf
###sed -ie 's/-f90=ifort//g' ./configure.wrf
fi

ulimit -s unlimited
./compile em_real 2>&1 |tee compile.log
EOF2

  job_id_install=$(sbatch --parsable install_wrf.slurm) 
  
  echo "install_wrf.slurm submitted with ID: $job_id_install"
  
  echo "Waiting for Installation ($job_id_install) to finish ...."
  
  while [ "$(squeue -h -j $job_id_install | wc -l )" -gt 0 ]; do
     sleep 10
  done
  
  job_status_install=$(sacct -j $job_id_install --format=State --noheader | awk '{print $1}' | uniq)
  
  if [[ "$job_status_install" != "COMPLETED" ]]; then
    echo "WARNING: Installation ($job_id_install) failed with status: $job_status_install"
    exit 1
  fi

fi

#--- END  OF RUN COMPILE WRF -----------------------------------------------------------------------------

#--- RUN COMPILE WPS -------------------------------------------------------------------------------------
if [[ $RUN_COMPILE_WPS -eq 1 ]]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         COMPILE WPS        '
echo
echo '-------------------------------------------------------------------------'
echo
cd $WPS_DIR
cat > install_wps.slurm << EOF2
#!/bin/bash
#SBATCH --job-name=CompileWPS               # nom du job
#SBATCH --partition=zen16                   # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=20GB                          # memory limit
#SBATCH --time=01:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CompileWRFChem%j.out       # nom du fichier de sortie
#SBATCH --error=CompileWRFChem%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

# Link the necessary files from the SI directory

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
echo \$PWD

export WPSDIR=\$PWD

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}
export MPI_LIB=-L\${OPENMPI_ROOT}/lib

export WRF_DIR=$WRF_DIR
export TOOLDIR=$HOME/tools_spirit
export PATH=\$TOOLDIR/bin:\$PATH
export PATH=\$TOOLDIR/lib:\$PATH
export WRFIO_NCD_NO_LARGE_FILE_SUPPORT=0
export JASPERLIB=\$JASPER_ROOT/lib
export JASPERINC=\$JASPER_ROOT/include
export WRF_EM_CORE=1
export WRF_NMM_CORE=0
export WRF_CHEM=1
export WRF_KPP=1
export YACC="\$TOOLDIR/bin/yacc -d"
export FLEX_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export HDF5_DISABLE_VERSION_CHECK=1


cd \$WPSDIR
./clean -a

###   21.  Linux x86_64, Intel Classic compilers    (serial)
###   22.  Linux x86_64, Intel Classic compilers    (serial_NO_GRIB2)
###   23.  Linux x86_64, Intel Classic compilers    (dmpar)
###   24.  Linux x86_64, Intel Classic compilers    (dmpar_NO_GRIB2)
./configure <<EOF
23

EOF

#sed -ie 's/hdf5hl_fortran/hdf5_hl_fortran/' ./configure.wrf

./compile  2>&1 |tee compile.log

EOF2
  
  job_id_install=$(sbatch --parsable install_wps.slurm) 
  
  echo "install_wps.slurm submitted with ID: $job_id_install"
  
  echo "Waiting for Installation ($job_id_install) to finish ...."
  
  while [ "$(squeue -h -j $job_id_install | wc -l )" -gt 0 ]; do
     sleep 10
  done
  
  job_status_install=$(sacct -j $job_id_install --format=State --noheader | awk '{print $1}' | uniq)
  
  if [[ "$job_status_install" != "COMPLETED" ]]; then
    echo "WARNING: Installation ($job_id_install) failed with status: $job_status_install"
    exit 1
  fi

fi

#--- END OF 'RUN COMPILE WPS' ----------------------------------------------------------------------------

#--- RUN WPS ---------------------------------------------------------------------------------------------

if [ $RUN_WPS -eq 1 ]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN WPS        '
echo
echo '-------------------------------------------------------------------------'
echo

# Check if namelist.wps already exists

WPSnamelist="$WPSdomain_DIR/namelist.wps"
if [ ! -e $WPSnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 0 ]; then
  echo "You set USE_DEFAULT_NAMELIST=0 (i.e. you want to use your own namelist.wps)"
  echo "But, namelist.wps does not exist in (\$WPSdomain_DIR:$WPSdomain_DIR)"
  echo "Prepare namelist.wps or use the default namelist (set USE_DEFAULT_NAMELIST=1)"
  exit 1
fi

if [ -e $WPSnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo "You set USE_DEFAULT_NAMELIST=1 (i.e. you want to use a default namelist.wps)"
  echo "But, we find a namelist.wps in $WPSdomain_DIR"
  echo "I backup your namelist.wps as namelist.wps.backup.$PID"
  mv -f $WPSdomain_DIR/namelist.wps $WPSdomain_DIR/namelist.wps.backup.$PID
fi

if [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo "namelist.wps file does not exist in $WPSdomain_DIR"
  echo "use a default namelist.wps"
  cp $LAUNCH_DIR/namelist.wps.default $WPSnamelist

  start_date=$STARTDATETIME
  echo "start date : $start_date"

  end_date=$ENDDATETIME
  echo "end date : $end_date"

  YYYYs=${start_date:0:4}
  MMs=${start_date:5:2}
  DDs=${start_date:8:2}
  HHs=${start_date:11:2}
  YYYYe=${end_date:0:4}
  MMe=${end_date:5:2}
  DDe=${end_date:8:2}
  HHe=${end_date:11:2}
  x2NN=$(echo "scale=1; $xNN / 2" | bc -l)
  y2NN=$(echo "scale=1; $yNN / 2" | bc -l)

  sed -i "s/MAP_PROJ/$map_proj/" $WPSnamelist
  sed -i "s/YYYYs/$YYYYs/" $WPSnamelist
  sed -i "s/MMs/$MMs/" $WPSnamelist
  sed -i "s/DDs/$DDs/" $WPSnamelist
  sed -i "s/HHs/$HHs/" $WPSnamelist
  sed -i "s/YYYYe/$YYYYe/" $WPSnamelist
  sed -i "s/MMe/$MMe/" $WPSnamelist
  sed -i "s/DDe/$DDe/" $WPSnamelist
  sed -i "s/HHe/$HHe/" $WPSnamelist
  sed -i "s/xNN/$xNN/" $WPSnamelist
  sed -i "s/yNN/$yNN/" $WPSnamelist
  sed -i "s/x2NN/$x2NN/" $WPSnamelist
  sed -i "s/y2NN/$y2NN/" $WPSnamelist
  sed -i "s/RefLat/$RefLat/" $WPSnamelist
  sed -i "s/RefLon/$RefLon/" $WPSnamelist
  sed -i "s/TrueLat1/$TrueLat1/" $WPSnamelist
  sed -i "s/TrueLat2/$TrueLat2/" $WPSnamelist
  sed -i "s/StandLon/$StandLon/" $WPSnamelist
  sed -i "s/DDx/$DDx/" $WPSnamelist
  sed -i "s/DDy/$DDx/" $WPSnamelist

  if cmp -s $WPSnamelist $WPSdomain_DIR/namelist.wps.backup.$PID; then
    echo "Default namelist.wps is identical to the backed up namelist.wps.backup.$PID."
    echo "Deleting namelist.wps.backup.$PID"
    rm -f "$WPSdomain_DIR/namelist.wps.backup.$PID"
  fi
fi

meteo_dir="/proju/wrf-chem/onishi/FNL/ds083.2/"
start_ts=$(date -d "$start_date" +%s)
end_ts=$(date -d "$end_date" +%s)

#----   WPS : test domain

if [ ! -d "$WPSdomain_DIR" ]; then
  mkdir -p $WPSdomain_DIR
fi
cd $WPSdomain_DIR

#-----------------------------------------------------
#
#  Link FNL files (exit if files are not available)
#
#----------------------------------------------------

meteo_step=$((3600*6))
missing_files=0
# loop over the time range
current_ts=$start_ts
while [ $current_ts -le $end_ts ]; do
    # Extract YYYY, MM, DD, hh from the timestamp
    YYYY=$(date -d "@$current_ts" +%Y)
    MM=$(date -d "@$current_ts" +%m)
    DD=$(date -d "@$current_ts" +%d)
    hh=$(date -d "@$current_ts" +%H)

    # Construct the expected file path
    file_path="$meteo_dir/FNL$YYYY/fnl_${YYYY}${MM}${DD}_${hh}_00.grib2"

    # Check if file exists
    if [ ! -e "$file_path" ]; then
        echo "Missing file: $file_path"
        missing_files=$((missing_files + 1))
    else
        ln -sf $file_path .
    fi

    # Move to the next hour
    current_ts=$((current_ts + meteo_step))
done

# Final summary
if [ $missing_files -eq 0 ]; then
    echo "✅ All files are available."
else
    echo "⚠️ $missing_files files are missing."
    exit 1
fi

cat > run_wps.slurm <<EOF2
#!/bin/bash
###SBATCH --account=wlm@cpu
#SBATCH --job-name=RunWPS           # nom du job
#SBATCH --partition=zen16                   # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
####SBATCH --cpus-per-task=5                   # 5 x 4GB = 20 GB memory
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=60GB                          # memory limit
#####SBATCH --hint=nomultithread                # 1 pocessur MPI par coeur physique
#SBATCH --time=04:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=WPS%j.out       # nom du fichier de sortie
#SBATCH --error=WPS%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

source /usr/share/modules/init/bash

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}
export HDF5_DISABLE_VERSION_CHECK=1


NETCDF_ROOT=\$NETCDF
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$NETCDF_ROOT/lib:/home/onishi/tools_spirit/lib"
ulimit -s unlimited
ulimit unlimited

# Run job

export WPSDIR=$WPS_DIR

ln -sf $WPS_DIR/metgrid.exe metgrid.exe
ln -sf $WPS_DIR/ungrib.exe ungrib.exe
ln -sf $WPS_DIR/geogrid.exe geogrid.exe
ln -sf $WPS_DIR/util/avg_tsfc.exe .
ln -sf $WPS_DIR/link_grib.csh .

ln -sf $WPS_DIR/ungrib/Variable_Tables/Vtable.GFS Vtable
ln -sf $WPS_DIR/metgrid/METGRID.TBL.ARW METGRID.TBL
ln -sf $WPS_DIR/geogrid/GEOGRID.TBL.ARW GEOGRID.TBL


./link_grib.csh ./fnl*

# Create a SHELL script which set stacksize unlimited

ulimit -s unlimited
mpirun ./geogrid.exe
mpirun ./ungrib.exe
#mpirun ./avg_tsfc.exe

rm met_em.d*

mpirun ./metgrid.exe

EOF2

#--- launch wps in batch mode ------

job_id_wps=$(sbatch --parsable run_wps.slurm) 

echo "run_wps.slurm submitted with ID: $job_id_wps"
echo "Waiting for WPS ($job_id_wps) to finish ...."

while [ "$(squeue -h -j $job_id_wps | wc -l )" -gt 0 ]; do
   sleep 10
done

job_status_wps=$(sacct -j $job_id_wps --format=State --noheader | awk '{print $1}' | uniq)

if [[ "$job_status_wps" != "COMPLETED" ]]; then
  echo "WARNING: WPS ($job_id_wps) failed with status: $job_status_wps"
  exit 1
else
  # Convert to seconds since epoch
  echo "start_date = $start_date"
  echo "end_date   = $end_date"
  start_sec=$(date -d "$start_date" +%s)
  end_sec=$(date -d "$end_date" +%s)
  
  all_exist=true
  
  for (( t=$start_sec; t<=$end_sec; t+=$meteo_step )); do
      filename="met_em.d01.$(date -d @$t +%Y-%m-%d_%H:%M:%S).nc"
      if [ ! -e "$filename" ]; then
          echo "Missing file: $filename"
          all_exist=false
      fi
  done
  
  if $all_exist; then
      echo "WPS is finished. All met_em files have been successfully created."
  else
      echo "WPS is finished. But, some met_em files are missing."
  fi 
fi


fi
#--- END OF 'RUN WPS' ----------------------------------------------------------------------


#--- RUN REAL NoCHEM -----------------------------------------------------------------------
if [ $RUN_REAL_NOCHEM -eq 1 ]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN REAL WITHOUT CHEMISTRY        '
echo
echo '-------------------------------------------------------------------------'
echo
cd $WRFrun_DIR


# LINK met_em files from $WPSdomain_DIR to $WRFrun_DIR
ln -sf $WPSdomain_DIR/met_em*.nc .

WRFnamelist="$WRFrun_DIR/namelist.input"
if [ ! -e $WRFnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 0 ]; then
  echo "You set USE_DEFAULT_NAMELIST=0 (i.e. you want to use your own namelist.input)"
  echo "But, namelist.input does not exist in (\$WRFrun_DIR:$WRFrun_DIR)"
  echo "Prepare namelist.input or use the default namelist (set USE_DEFAULT_NAMELIST=1)"
  exit 1
fi

if [ -e $WRFnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo "You set USE_DEFAULT_NAMELIST=1 (i.e. you want to use a default namelist.input)"
  echo "But, we find a namelist.input in $WRFrun_DIR"
  echo "I backup your namelist.input as namelist.input.backup.$PID"
  mv -f $WRFrun_DIR/namelist.input $WRFrun_DIR/namelist.input.backup.$PID
fi

if [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo " a default namelist.input.default"
  cp $LAUNCH_DIR/namelist.input.default $WRFnamelist

  start_date=$STARTDATETIME
  echo "start date : $start_date"

  end_date=$ENDDATETIME
  echo "end date : $end_date"

  YYYYs=${start_date:0:4}
  MMs=${start_date:5:2}
  DDs=${start_date:8:2}
  HHs=${start_date:11:2}
  YYYYe=${end_date:0:4}
  MMe=${end_date:5:2}
  DDe=${end_date:8:2}
  HHe=${end_date:11:2}
  x2NN=$(echo "scale=1; $xNN / 2" | bc -l)
  y2NN=$(echo "scale=1; $yNN / 2" | bc -l)

  sed -i "s/YYYYs/$YYYYs/" $WRFnamelist
  sed -i "s/MMs/$MMs/" $WRFnamelist
  sed -i "s/DDs/$DDs/" $WRFnamelist
  sed -i "s/HHs/$HHs/" $WRFnamelist
  sed -i "s/YYYYe/$YYYYe/" $WRFnamelist
  sed -i "s/MMe/$MMe/" $WRFnamelist
  sed -i "s/DDe/$DDe/" $WRFnamelist
  sed -i "s/HHe/$HHe/" $WRFnamelist
  sed -i "s/xNN/$xNN/" $WRFnamelist
  sed -i "s/yNN/$yNN/" $WRFnamelist
  sed -i "s/x2NN/$x2NN/" $WRFnamelist
  sed -i "s/y2NN/$y2NN/" $WRFnamelist
  sed -i "s/RefLat/$RefLat/" $WRFnamelist
  sed -i "s/RefLon/$RefLon/" $WRFnamelist
  sed -i "s/DDx/$DDx/" $WRFnamelist
  sed -i "s/DDy/$DDx/" $WRFnamelist
  sed -i "s/TIMESTEP/$TIMESTEP/" $WRFnamelist
  sed -i "s/CHEMOPT/0/" $WRFnamelist

  if cmp -s $WRFnamelist $WRFrun_DIR/namelist.input.backup.$PID; then
    echo "Default namelist.input is identical to the backed up namelist.input.backup.$PID."
    echo "Deleting namelist.input.backup.$PID"
    rm -f "$WRFrun_DIR/namelist.input.backup.$PID"
  fi
fi

cat > run_real.slurm << EOF2
#!/bin/bash

#SBATCH --job-name=Real           # nom du job
#SBATCH --partition=zen16         # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=4                # nombre de taches
#SBATCH --ntasks-per-node=4       # nombre de taches MPI par noeud
#SBATCH --mem=50GB                # memory limit
#SBATCH --time=04:00:00           # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=Real%j.out       # nom du fichier de sortie
#SBATCH --error=Real%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}

export MPI_LIB=-L\${OPENMPI_ROOT}/lib
export HDF5_DISABLE_VERSION_CHECK=1

for file in \`ls ${WRF_DIR}/run | grep -v namelist\`
do
  echo \$file
  ln -sf ${WRF_DIR}/run/\${file} .
done

ln -sf $WRF_DIR/main/real.exe .

ulimit -s unlimited
mpirun ./real.exe
EOF2

  job_id=$(sbatch --parsable run_real.slurm) 
  
  echo "run_real.slurm submitted with ID: $job_id"
  
  echo "Waiting for real.exe ($job_id) to finish ...."
  
  while [ "$(squeue -h -j $job_id | wc -l )" -gt 0 ]; do
     sleep 10
  done
  
  job_status=$(sacct -j $job_id --format=State --noheader | awk '{print $1}' | uniq)
  
  if [[ "$job_status" != "COMPLETED" ]]; then
    echo "WARNING: real.exe ($job_id) failed with status: $job_status"
    exit 1
  else
    if tail -n 20 rsl.out.0000 | grep -q "SUCCESS"; then
      echo "real.exe without chemistry is successfully complete"
    fi
  fi

fi
#--- END OF RUN REAL NoCHEM ----------------------------------------------------------------

#--- RUN MEGAN -----------------------------------------------------------------------------

if [[ $RUN_MEGAN -eq 1 ]]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN MEGAN        '
echo
echo '-------------------------------------------------------------------------'
echo

if [[ ! -e $MEGAN_exe ]]; then
  echo "MEGAN executable (megan_bio_emiss) can not be found."
  echo "Compile and assign it to 'megan_exe' variable in 'USER INPUT'."
  exit 1
fi

if [ ! -d "$MEGAN_DIR" ]; then
  mkdir -p $MEGAN_DIR
fi

cd $MEGAN_DIR

start_date=$STARTDATETIME
end_date=$ENDDATETIME
MMs=$(echo ${start_date:5:2} | sed 's/^0*//')
MMe=$(echo ${end_date:5:2}   | sed 's/^0*//')
if (( MMs > MMe )); then
  MMs=1
  MMe=12
fi
#
# Create MEGAN inp file
#
cat > megan_bio_emiss.inp << EOF
&control

domains = 1,
start_lai_mnth =$MMs,
end_lai_mnth   =$MMe,
wrf_dir   = '$WRFrun_DIR'
megan_dir = '$MEGAN_data_DIR'

/
EOF

#
# Create SLURM batch file for MEGAN
#
cat > run_megan_bio_emiss.slurm <<EOF2
#!/bin/bash

#SBATCH --job-name=MEGAN                    # nom du job
#SBATCH --partition=zen16                   # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=20GB                          # memory limit
#SBATCH --time=04:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=Megan%j.out       # nom du fichier de sortie
#SBATCH --error=Megan%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}

export NETCDF_DIR=\$NETCDF

rm Megan*.out

# Run job
${MEGAN_exe} < megan_bio_emiss.inp 2>&1 |tee megan_bio_emiss.out

EOF2

job_id=$(sbatch --parsable run_megan_bio_emiss.slurm) 

echo "run_megan_bio_emiss.slurm submitted with ID: $job_id"

echo "Waiting for MEGAN ($job_id) to finish ...."

while [ "$(squeue -h -j $job_id | wc -l )" -gt 0 ]; do
   sleep 10
done

job_status=$(sacct -j $job_id --format=State --noheader | awk '{print $1}' | uniq)

if [[ "$job_status" != "COMPLETED" ]]; then
  echo "WARNING: MEGAN ($job_id) failed with status: $job_status"
  exit 1
else
  if [ -e ./wrfbiochemi_d01 ]; then
    echo "MEGAN: wrfbiochemi_d01 is successfully created"
    echo "Copying wrfbiochemi_d01 to $WRFrun_DIR"
    cp -f ./wrfbiochemi_d01 $WRFrun_DIR/.
  else
    echo "ERROR: MEGAN: wrfbiochemi_d01 is not created."
    exit 1 
  fi
fi


fi

#--- END OF RUN MEGAN -----------------------------------------------------------------------------

#--- RUN Wesely and exo_coldens --------------------------------------------------------------

if [[ $RUN_WESELY_EXO_COLDENS -eq 1 ]]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN WESELY and EXO_COLDENS        '
echo
echo '-------------------------------------------------------------------------'
echo

cd $WESCOL_DIR

if [[ ! -e $wesely_exe || ! -e $exo_coldens_exe ]]; then
  echo "wesely and/or exo_coldens_exe can not be found."
  echo "Compile and assign them to 'wesely_exe' and 'exo_coldens_exe' variables in 'USER INPUT'."
  exit 1
fi

### if [ ! -d "$WESCOL_DIR" ]; then
###   mkdir -p $WESCOL_DIR 
### fi cd $WESCOL_DIR

#
# Create wesely.input file
#
cat > wesely.input << EOF
&control

wrf_dir = '$WRFrun_DIR'
domains = 1,

/
EOF
#
# Create exo_coldens.input file
#
cat > exo_coldens.input << EOF
&control

wrf_dir = '$WRFrun_DIR'
domains = 1,

/
EOF
#
# Run wesely and exo_coldens
#
module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 
$wesely_exe < wesely.input > wesely.out
$exo_coldens_exe < exo_coldens.input > exo_coldens.out


if tail -n 3 wesely.out | grep -q "completed successfully"; then
  echo "wesely finished successfully.. "
  echo "Copying wrf_season_wes_usgs_d01.nc to $WRFrun_DIR"
  cp -f ./wrf_season_wes_usgs_d01.nc $WRFrun_DIR/.
else
  echo "wesely did not finish successfully..."
  exit 1
fi

if tail -n 3 exo_coldens.out | grep -q "completed successfully"; then
  echo "exo_coldens finished successfully.. "
  echo "Copying exo_coldens_d01 to $WRFrun_DIR"
  cp -f ./exo_coldens_d01 $WRFrun_DIR/.
else
  echo "exo_coldens did not finish successfully..."
  exit 1
fi

fi

#--- END OF RUN Wesely and exo_coldens --------------------------------------------------------------

#--- RUN REAL with CHEM -----------------------------------------------------------------------
if [ $RUN_REAL_CHEM -eq 1 ]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN REAL with CHEMISTRY        '
echo
echo '-------------------------------------------------------------------------'
echo
cd $WRFrun_DIR


# LINK met_em files from $WPSdomain_DIR to $WRFrun_DIR
ln -sf $WPSdomain_DIR/met_em*.nc .

WRFnamelist="$WRFrun_DIR/namelist.input"
if [ ! -e $WRFnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 0 ]; then
  echo "You set USE_DEFAULT_NAMELIST=0 (i.e. you want to use your own namelist.input)"
  echo "But, namelist.input does not exist in (\$WRFrun_DIR:$WRFrun_DIR)"
  echo "Prepare namelist.input or use the default namelist (set USE_DEFAULT_NAMELIST=1)"
  exit 1
fi

if [ -e $WRFnamelist ] && [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo "You set USE_DEFAULT_NAMELIST=1 (i.e. you want to use a default namelist.input)"
  echo "But, we find a namelist.input in $WRFrun_DIR"
  echo "I backup your namelist.input as namelist.input.backup.$PID"
  mv -f $WRFrun_DIR/namelist.input $WRFrun_DIR/namelist.input.backup.$PID
fi

if [ $USE_DEFAULT_NAMELIST -eq 1 ]; then
  echo " a default namelist.input.default"
  cp $LAUNCH_DIR/namelist.input.default $WRFnamelist

  start_date=$STARTDATETIME
  echo "start date : $start_date"

  end_date=$ENDDATETIME
  echo "end date : $end_date"

  YYYYs=${start_date:0:4}
  MMs=${start_date:5:2}
  DDs=${start_date:8:2}
  HHs=${start_date:11:2}
  YYYYe=${end_date:0:4}
  MMe=${end_date:5:2}
  DDe=${end_date:8:2}
  HHe=${end_date:11:2}
  x2NN=$(echo "scale=1; $xNN / 2" | bc -l)
  y2NN=$(echo "scale=1; $yNN / 2" | bc -l)

  sed -i "s/YYYYs/$YYYYs/" $WRFnamelist
  sed -i "s/MMs/$MMs/" $WRFnamelist
  sed -i "s/DDs/$DDs/" $WRFnamelist
  sed -i "s/HHs/$HHs/" $WRFnamelist
  sed -i "s/YYYYe/$YYYYe/" $WRFnamelist
  sed -i "s/MMe/$MMe/" $WRFnamelist
  sed -i "s/DDe/$DDe/" $WRFnamelist
  sed -i "s/HHe/$HHe/" $WRFnamelist
  sed -i "s/xNN/$xNN/" $WRFnamelist
  sed -i "s/yNN/$yNN/" $WRFnamelist
  sed -i "s/x2NN/$x2NN/" $WRFnamelist
  sed -i "s/y2NN/$y2NN/" $WRFnamelist
  sed -i "s/RefLat/$RefLat/" $WRFnamelist
  sed -i "s/RefLon/$RefLon/" $WRFnamelist
  sed -i "s/DDx/$DDx/" $WRFnamelist
  sed -i "s/DDy/$DDx/" $WRFnamelist
  sed -i "s/TIMESTEP/$TIMESTEP/" $WRFnamelist
  sed -i "s/CHEMOPT/$CHEMOPT/" $WRFnamelist

  if cmp -s $WRFnamelist $WRFrun_DIR/namelist.input.backup.$PID; then
    echo "Default namelist.input is identical to the backed up namelist.input.backup.$PID."
    echo "Deleting namelist.input.backup.$PID"
    rm -f "$WRFrun_DIR/namelist.input.backup.$PID"
  fi
fi

cat > run_real.slurm << EOF2
#!/bin/bash

#SBATCH --job-name=Real           # nom du job
#SBATCH --partition=zen16         # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=4                # nombre de taches
#SBATCH --ntasks-per-node=4       # nombre de taches MPI par noeud
#SBATCH --mem=50GB                # memory limit
#SBATCH --time=04:00:00           # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=Real%j.out       # nom du fichier de sortie
#SBATCH --error=Real%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}/
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}

export MPI_LIB=-L\${OPENMPI_ROOT}/lib
export HDF5_DISABLE_VERSION_CHECK=1

for file in \`ls ${WRF_DIR}/run | grep -v namelist\`
do
  echo \$file
  ln -sf ${WRF_DIR}/run/\${file} .
done

ln -sf $WRF_DIR/main/real.exe .

ulimit -s unlimited
mpirun ./real.exe
EOF2

  job_id=$(sbatch --parsable run_real.slurm) 
  
  echo "run_real.slurm submitted with ID: $job_id"
  
  echo "Waiting for real.exe ($job_id) to finish ...."
  
  while [ "$(squeue -h -j $job_id | wc -l )" -gt 0 ]; do
     sleep 10
  done
  
  job_status=$(sacct -j $job_id --format=State --noheader | awk '{print $1}' | uniq)
  
  if [[ "$job_status" != "COMPLETED" ]]; then
    echo "WARNING: real.exe ($job_id) failed with status: $job_status"
    exit 1
  else
    if tail -n 20 rsl.out.0000 | grep -q "SUCCESS"; then
      echo "real.exe without chemistry is successfully complete"
    fi
  fi

fi

#--- END OF RUN REAL with CHEM -----------------------------------------------------------------------

#--- RUN MOZBC ------------------------------------------------------------------------------------
if [[ $RUN_MOZBC -eq 1 ]]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN MOZBC        '
echo
echo '-------------------------------------------------------------------------'
echo

echo ${MOZBC_DIR}
if [ ! -d ${MOZBC_DIR} ]; then
  mkdir -p ${MOZBC_DIR}
fi
cd $MOZBC_DIR
pwd
cat > run_mozbc.slurm <<EOF2
#!/bin/bash

###SBATCH --account=wlm@cpu
#SBATCH --job-name=MOZBC           # nom du job
#SBATCH --partition=zen16                  # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=60GB                          # memory limit
#SBATCH --time=04:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=mozbc%j.out       # nom du fichier de sortie
#SBATCH --error=mozbc%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 

export NETCDF=$NETCDF_DIR

# Run job
${mozbc_exe} < cesm.TEST.MOZARTMOSAIC.inp  > mozbc.out
EOF2

cat > cesm.TEST.MOZARTMOSAIC.inp <<EOF2
&control
do_bc     = .true.
do_ic     = .true.
domain    = 1,
dir_wrf   = '${WRFrun_DIR}'
dir_moz   = '$(dirname "$mozbc_data")'
fn_moz    = '$(basename "$mozbc_data")'
moz_var_suffix = ''
def_missing_var = .true.

spc_map = 'o3 -> O3', 'n2o -> N2O', 'no -> NO',
          'no2 -> NO2', 'nh3 -> NH3', 'hno3 -> HNO3', 'hno4 -> HO2NO2',
          'n2o5 -> N2O5', 'h2o2 -> H2O2',
          'ch4 -> CH4', 'co -> CO', 'ch3ooh -> CH3OOH',
          'hcho -> CH2O', 'ch3oh -> CH3OH', 'c2h4 -> C2H4',
          'ald -> CH3CHO', 'acet -> CH3COCH3', 'mgly -> CH3COCHO',
          'pan -> PAN', 'mpan -> MPAN', 'macr -> MACR',
          'mvk -> MVK', 'c2h6 -> C2H6', 'c3h6 -> C3H6', 'c3h8 -> C3H8',
          'c2h5oh -> C2H5OH', 'c10h16 -> MTERP',
          'isopr -> ISOP','acetol -> HYAC', 'mek -> MEK',
          'bigene -> BIGENE', 'bigalk -> BIGALK',
          'tol -> TOLUENE', 'benzene -> BENZENE', 'xylenes -> XYLENES',
          'cres -> CRESOL', 'dms -> DMS', 'so2 -> SO2',
          'oc_a01 -> 0.9886*soa_a2+0.1216*soa_a1+0.1123*pom_a1;1.0e9',
          'oc_a02 -> 0.0114*soa_a2+0.7618*soa_a1+0.3783*pom_a1;1.0e9',
          'oc_a03 -> 0.0000*soa_a2+0.1164*soa_a1+0.0087*pom_a1;1.0e9',
          'oc_a04 -> 0.0000*soa_a2+0.0002*soa_a1+0.0000*pom_a1;1.0e9',
          'bc_a01->0.1216*bc_a1+0.1216*bc_a4;1.e9',
          'bc_a02->0.7618*bc_a1+0.7618*bc_a4;1.e9',
          'bc_a03->0.1164*bc_a1+0.1164*bc_a4;1.e9',
          'bc_a04->0.0002*bc_a1+0.0002*bc_a4;1.e9',
          'so4_a01->0.9886*so4_a2+0.1216*so4_a1+0.0000*so4_a3;1.e9',
          'so4_a02->0.0114*so4_a2+0.7618*so4_a1+0.0002*so4_a3;1.e9',
          'so4_a03->0.0000*so4_a2+0.1164*so4_a1+0.0995*so4_a3;1.e9',
          'so4_a04->0.0000*so4_a2+0.0002*so4_a1+0.9003*so4_a3;1.e9',
          'nh4_a01->0.1856*so4_a2+0.0050*so4_a1+0.0000*so4_a3;1.e9',
          'nh4_a02->0.0021*so4_a2+0.0930*so4_a1+0.0000*so4_a3;1.e9',
          'nh4_a03->0.0000*so4_a2+0.0203*so4_a1+0.0186*so4_a3;1.e9',
          'nh4_a04->0.0000*so4_a2+0.0000*so4_a1+0.1690*so4_a3;1.e9',
          'no3_a01->0.0000*so4_a2+0.0000*so4_a1+0.0000*so4_a3;1.e9',
          'no3_a02->0.0000*so4_a2+0.0000*so4_a1+0.0000*so4_a3;1.e9',
          'no3_a03->0.0000*so4_a2+0.0000*so4_a1+0.0000*so4_a3;1.e9',
          'no3_a04->0.0000*so4_a2+0.0000*so4_a1+0.0000*so4_a3;1.e9',
          'na_a01->0.3889*ncl_a2+0.0479*ncl_a1+0.0000*ncl_a3;1.e9',
          'na_a02->0.0045*ncl_a2+0.2997*ncl_a1+0.0000*ncl_a3;1.e9',
          'na_a03->0.0000*ncl_a2+0.0458*ncl_a1+0.0391*ncl_a3;1.e9',
          'na_a04->0.0000*ncl_a2+0.0000*ncl_a1+0.3542*ncl_a3;1.e9',
          'cl_a01->0.5996*ncl_a2+0.0737*ncl_a1+0.0000*ncl_a3;1.e9',
          'cl_a02->0.0068*ncl_a2+0.4621*ncl_a1+0.0000*ncl_a3;1.e9',
          'cl_a03->0.0000*ncl_a2+0.0709*ncl_a1+0.0604*ncl_a3;1.e9',
          'cl_a04->0.0000*ncl_a2+0.0001*ncl_a1+0.5462*ncl_a3;1.e9',
          'oin_a01->0.9886*dst_a2+0.1216*dst_a1+0.0000*dst_a3;1.e9',
          'oin_a02->0.0114*dst_a2+0.7618*dst_a1+0.0002*dst_a3;1.e9',
          'oin_a03->0.0000*dst_a2+0.1164*dst_a1+0.0995*dst_a3;1.e9',
          'oin_a04->0.0000*dst_a2+0.0002*dst_a1+0.9003*dst_a3;1.e9',
          'num_a01->0.9996*num_a2+0.7135*num_a1+0.0000*num_a3;1.0',
          'num_a02->0.0004*num_a2+0.2847*num_a1+0.0239*num_a3;1.0',
          'num_a03->0.0000*num_a2+0.0016*num_a1+0.6258*num_a3;1.0',
          'num_a04->0.0000*num_a2+0.0000*num_a1+0.3501*num_a3;1.0',
/
EOF2

job_id_mozbc=$(sbatch --parsable run_mozbc.slurm) 

echo "run_mozbc.slurm submitted with ID: $job_id_mozbc"

echo "Waiting for mozbc ($job_id_mozbc) to finish ...."

while [ "$(squeue -h -j $job_id_mozbc | wc -l )" -gt 0 ]; do
   sleep 10
done

job_status_mozbc=$(sacct -j $job_id_mozbc --format=State --noheader | awk '{print $1}' | uniq)

if [[ "$job_status_mozbc" != "COMPLETED" ]]; then
  echo "WARNING: MOZBC ($job_id_mozbc) failed with status: $job_status_mozbc"
  exit 1
else
  if tail -n 5 mozbc.out | grep -q "completed successfully"; then
    echo "mozbc completed successfully"
  fi 
fi

fi




#--- END OF RUN MOZBC ------------------------------------------------------------------------------------

#--- PREPARE WRFCHEMI INPUT FILES --------------------------------------------------------------
if [[ $RUN_WRFCHEMI -eq 1 ]]; then

echo
echo '-------------------------------------------------------------------------'
echo
echo '         CREATE WRFCHEMI        '
echo
echo '-------------------------------------------------------------------------'
echo
cd $LAUNCH_DIR

if [ ! -e ${wrfchemi_python} ]; then
  echo "You should have ${wrfchemi_python} in ${LAUNCH_DIR}"
  exit 1
fi 

sed "s|<WRFrun_DIR>|\"${WRFrun_DIR}\"|" ${wrfchemi_python}    > ./python_temp.py3
sed -i "s|<wrfchemi_DIR>|\"${wrfchemi_DIR}\"|" ./python_temp.py3

cat > run_create_wrfchemi.slurm <<EOF2
#!/bin/bash

#SBATCH --job-name=PythonEmission           # nom du job
#SBATCH --partition=zen4                    # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=10GB                          # memory limit
#SBATCH --time=24:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=PythonEmission%j.out       # nom du fichier de sortie
#SBATCH --error=PythonEmission%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load python/meso-3.9
### conda activate my_python3
### module load netcdf4/4.3.3.1-ifort
### export PROJ_LIB='/home/onishi/.conda/pkgs/proj4-5.2.0-h14c3975_1001/share/proj/'

### rm PythonEmission*.out

which python
mpirun python ./python_temp.py3
EOF2

job_id=$(sbatch --parsable run_create_wrfchemi.slurm) 

echo "run_create_wrfchemi.slurm submitted with ID: $job_id"

echo "Waiting for create_wrfchemi ($job_id) to finish ...."

while [ "$(squeue -h -j $job_id | wc -l )" -gt 0 ]; do
   sleep 10
done

job_status=$(sacct -j $job_id --format=State --noheader | awk '{print $1}' | uniq)

if [[ "$job_status" != "COMPLETED" ]]; then
  echo "WARNING: create_wrfchemi ($job_id) failed with status: $job_status"
  exit 1
### else
###   if [ -e ./wrfbiochemi_d01 ]; then
###     echo "MEGAN: wrfbiochemi_d01 is successfully created"
###     echo "Copying wrfbiochemi_d01 to $WRFrun_DIR"
###     cp -f ./wrfbiochemi_d01 $WRFrun_DIR/.
###   else
###     echo "ERROR: MEGAN: wrfbiochemi_d01 is not created."
###     exit 1 
###   fi
fi

if tail -n 1 ./PythonEmission${job_id}.out | grep -q "Done"; then
  echo "create_wrfchemi.py3 completed"

  # Convert to seconds since epoch
  start_sec=$(date -d "$STARTDATETIME" +%s)
  end_sec=$(date -d "$ENDDATETIME" +%s)
 
  missing_files=0
  t=$start_sec
  
  while [ $t -le $end_sec ]; do
    # Format datetime
    file_datetime=$(date -d "@$t" +"%Y-%m-%d_%H:00:00")
    filename="${wrfchemi_DIR}/wrfchemi_d01_${file_datetime}"
  
    # Check file existence
    if [ ! -e "$filename" ]; then
        echo "Missing file: $filename"
        missing_files=$((missing_files + 1))
    fi
  
    # Advance by 1 hour (3600 seconds)
    t=$((t + 3600))
  done
  
  # Final message
  if [ $missing_files -eq 0 ]; then
      echo "All wrfchemi files are present."
  else
      echo "$missing_files files are missing."
      exit 1
  fi
fi

if [[ $WRF_CHEM -eq 1 ]]; then
  cd $WRFrun_DIR
  ln -sf ${wrfchemi_DIR}/wrfchemi_d01* .
fi

fi

#--- END OF PREPARE WRFCHEMI INPUT FILES --------------------------------------------------------------

#--- RUN WRF -----------------------------------------------------------

if [[ $RUN_WRF -eq 1 ]]; then
echo
echo '-------------------------------------------------------------------------'
echo
echo '         RUN WRF                '
echo
echo '-------------------------------------------------------------------------'
echo

cd $WRFrun_DIR

if [ ! -e "./wrfbdy_d01" ] || \
   [ ! -e "./wrfinput_d01" ] || \
   [ ! -e "./wrflowinp_d01" ] || \
   [ ! -e "./wrffdda_d01" ]; then
  
  echo "You need to run real.exe first"
  echo "With chemistry, you also need to run mozbc"
  echo "Set WRF_REAL_CHEM=1 and RUN_MOZBC=1"
  echo "And restart this script"
  exit 1
fi

if [ ! -e "./wrf_season_wes_usgs_d01.nc" ] || \
   [ ! -e "./exo_coldens_d01" ]; then
  echo "You need to run wesely and exo_coldens"
  echo "Set RUN_WESELY_EXO_COLDENS=1 and restart this script."
  exit 1
fi

if [ ! -e "./wrfbiochemi_d01" ]; then
  echo "You need to run MEGAN"
  echo "Set RUN_MEGAN=1 and restart this script"
  exit 1
fi

cat > run_wrf.slurm <<EOF2
#!/bin/bash

#SBATCH --job-name=runWRFChem           # nom du job
#SBATCH --partition=zen4                      # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=16                         # nombre de taches
#SBATCH --ntasks-per-node=16               # nombre de taches MPI par noeud
#SBATCH --mem=50GB                          # memory limit
#SBATCH --time=02:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=WRF%j.out       # nom du fichier de sortie
#SBATCH --error=WRF%j.error.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load  $module_intel  
module load  $module_openmpi
module load  $module_netcdfc
module load  $module_netcdff
module load  $module_hdf5   
module load  $module_jasper 
export CFLAGS="-I\${OPENMPI_ROOT}/include -m64"
export LDFLAGS="-L\${OPENMPI_ROOT}/lib -lmpi"
export NETCDF=${NETCDF_DIR}/
export PHDF5=\${HDF5_ROOT}
export HDF5=\${HDF5_ROOT}

export MPI_LIB=-L\${OPENMPI_ROOT}/lib
export HDF5_DISABLE_VERSION_CHECK=1

# Create a SHELL script which set stacksize unlimited
# and run wrf.exe

for file in \`ls ${WRF_DIR}/run | grep -v namelist\`
do
  echo \$file
  ln -sf ${WRF_DIR}/run/\${file} .
done

ln -sf $WRF_DIR/main/wrf.exe .

ulimit -s unlimited
mpirun ./wrf.exe
EOF2

  job_id=$(sbatch --parsable run_wrf.slurm) 
  
  echo "run_wrf.slurm submitted with ID: $job_id"
  
  echo "Waiting for real.exe ($job_id) to finish ...."
  
  while [ "$(squeue -h -j $job_id | wc -l )" -gt 0 ]; do
     sleep 10
  done
  
  job_status=$(sacct -j $job_id --format=State --noheader | awk '{print $1}' | uniq)
  
  if [[ "$job_status" != "COMPLETED" ]]; then
    echo "WARNING: wrf.exe ($job_id) failed with status: $job_status"
    exit 1
  else
    if tail -n 20 rsl.out.0000 | grep -q "SUCCESS"; then
      echo "wrf.exe is successfully complete"
    fi
  fi

fi



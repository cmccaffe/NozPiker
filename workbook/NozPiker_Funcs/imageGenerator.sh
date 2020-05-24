

resolution=$1
pdb=$2
moleculename=$3

pdb2mrc ${pdb}.pdb ${moleculename}_${pdb}_${resolution}.mrc apix=1 res=${resolution}


relion_project --i ${moleculename}_${pdb}_${resolution}.mrc --o ${moleculename}_${pdb}_${resolution}_proj --nr_uniform 30

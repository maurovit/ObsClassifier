REPOSITORY GITHUB  
https://github.com/maurovit/ObsClassifier.git

INSTALLAZIONE ED USO

Per poter utilizzare il software è necessario inserire nel folder 'softwares' un file CSV contenente la traduzione in valori di features 
delle classi del software da analizzare e, in ObserverClassifierTester.py, sostituire in SW_PATH 'TestSoftware' con il nome del nuovo file,
specificando '.csv' come sua estensione. Le combinazioni da testare come istanze del pattern observer, con le relative colonne features 
vuote, vengono salvate in un file CSV il cui percorso è specificato dalla variabile INSTANCES_COMBINATIONS_FILE_PATH in 
ObserverClassifierTester.py. Allo stato attuale, il classificatore InstancesClassifier effettua predizioni su features contenute in un file 
CSV indirizzato dalla varabile INSTANCES_MOKUP_PATH, usata nell'invocazione del metodo 'predict' di quest'ultimo in 
ObserverClassifierTester.py:  è preferibile sostituire quest'ultima con INSTANCES_COMBINATIONS_FILE_PATH, il cui file puntato deve essere 
avvalorato da un parser capace di computare le features o manualmente. Dopo la predizione dei ruoli e lagenerazione delle combinazioni, il 
software si ferma e richiede un input per procedere: tale operazione viene eseguita per permettere di avvalorare, eventualmente a mano, 
le features del file indicato da INSTANCES_COMBINATIONS_FILE_PATH.
I risultati di tutte le fasi vengono resitutiti nel folder 'predictions'. Per avviare il software è necessario eseguire 
ObserverClassifierTester.py.

#  Caso di studio A (ZeMA) — aggiornamento dell'analisi: budget di incertezza, dataset e split

Riassumo le modifiche apportate negli ultimi giorni (ottobre 2026) all'analisi del caso di studio A (stima della RUL sul cilindro elettromeccanico ZeMA). Il dettaglio è nei report del repository, che cito tra parentesi.

**1. Analisi approfondita del dataset** (`case_study_A_dataset_report.md`)

Ho confrontato il file pubblicato su Zenodo con Dorst et al. 2022. Struttura, sensori, selezione dei cicli e frequenze coincidono con il paper; i punti rilevanti sono:

- le incertezze dichiarate per la ZeMA DAQ non sono descritte nel paper: sono una costante per canale oppure esattamente l'1% / 1.5% della lettura, quindi valori di **tipo B** da datasheet. Di conseguenza anche *u*_physical è di tipo B (nella bozza era indicata come tipo A);
- per la pressione pneumatica l'incertezza dichiarata è circa 9 volte la variabilità tra cicli (*u/s* ≈ 8.8), per la velocità circa 4 volte, per l'accelerometro del cuscinetto liscio circa 2 volte;
- la libreria `zema_emc_annotated` con `datapoints_per_cycle=100` prende i **primi 100 campioni (50 ms)** della finestra di 1 s, non 100 campioni distribuiti: va dichiarato nel paper. Ho provato anche ingressi estratti dall'intera finestra (decimazione, medie a blocchi): l'accuratezza peggiora, quindi resta l'ingresso dei primi 50 ms;
- anomalie minori nei metadati (4776 cicli dichiarati contro 4766 effettivi) e incertezze duplicate tra grandezze diverse nella SmartUp Unit, che comunque non usiamo.

**2. Nuovo budget dell'incertezza** (`case_study_A_report_budget.md`)

Il budget originale propagava tutta la *u* dichiarata come rumore indipendente per lettura. I dati lo escludono: se *u* > *s*, la maggior parte di *u* deve essere comune a tutti i cicli, cioè sistematica. Una componente sistematica costante si cancella nella normalizzazione z-score quando training e inferenza usano lo stesso strumento (verificato numericamente). La perturbazione di circa 9 deviazioni standard portava inoltre la rete fuori dal dominio di training, e da qui veniva il disaccordo LPU/MC del 36%.

Il nuovo budget:

- separa *u* in parte casuale e sistematica (*u*² = *u*_rand² + *u*_sys²). *u*_rand è limitata superiormente dai dati (spread tra cicli, rumore tra campioni adiacenti) e, dove esiste, dal rumore di fondo del datasheet (Kistler 8704B50M1);
- per l'uso con lo stesso strumento propaga solo *u*_rand e aggiunge un termine *u*_calibration stimato su un validation set (15% del training): *u*_c = √(*u*_rand² + *u*_training² + *u*_calibration²), con Monte Carlo come stima primaria;
- riporta separatamente *u*_sys, che conta solo se il modello viene trasferito su un altro strumento.

Risultati (953 cicli di test, modello `datasheet`):

| | Budget originale | Budget corretto |
|---|---|---|
| *u*_c (stesso strumento) | 0.985 | **0.093** |
| LPU vs MC (mediana) | 36.5% | **3.6%** |
| Copertura k=2 / k=1 | 100% / – | 99.6% / 89.2% |
| *u*_c / MAE | ≈ 21 | ≈ 2 |
| corr(*u*_c, \|errore\|) | 0.26 | 0.45 |
| *u*_c per trasferimento (con *u*_sys) | – | 0.79 |

RMSE 0.066 e R² 0.95 restano praticamente invariati. L'incertezza diventa dello stesso ordine dell'errore reale e la propagazione al primo ordine torna accettabile. Rimane una sovracopertura (intervalli circa 2 volte troppo larghi). È dovuta alla parte casuale di velocità e accelerometro del cuscinetto liscio, per cui non esiste una specifica di rumore: la segnalerei come limite del dataset, che dichiara solo una *u* totale. Il termine *u*_calibration risulta sempre 0. Con il budget corretto, togliere il canale pressione non cambia più *u*_c per lo stesso strumento; dimezza solo la parte per il trasferimento.

**3. Nuova strategia di split**

L'etichetta dipende dall'indice del ciclo (*y* = *k*/*N*) e cicli consecutivi sono quasi identici. Con lo split casuale il 96% dei cicli di test ha un vicino nel training, quindi c'era il rischio di misurare un'interpolazione. Ho aggiunto uno **split a blocchi**: blocchi di 20, 50 o 100 cicli consecutivi assegnati a test/validation/training, con un gap di 5 cicli. Nessun ciclo di test ha più un vicino nel training. L'RMSE sale da 0.066 a 0.070–0.077 e R² resta 0.94–0.95: il leakage è piccolo e le conclusioni sul budget non cambiano (copertura k=2 99.3–99.9%, *u*_calibration = 0). Per i numeri principali si può quindi mantenere lo split casuale, citando quello a blocchi come controllo.

**4. Documenti e paper**

Ho aggiornato i report del repository (report principale, guida, README) con il budget corretto, il tipo B di *u*_physical e la notazione *u*_calibration. In `paper_revision_proposal.md` c'è una proposta di revisione della bozza, sezione per sezione, con il testo LaTeX sostitutivo. Riguarda soprattutto: Sezione 5 riscritta con i nuovi numeri, nuova open question sull'annotazione dell'incertezza nei dataset (separare componenti casuali e sistematiche) e conclusioni sulla validità dell'ipotesi di linearità.

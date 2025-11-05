# Sistema di Raccomandazione Medicinali Basato su AI

Un'applicazione CLI intelligente che utilizza il Machine Learning per suggerire il medicinale più appropriato (Paracetamolo o Ibuprofene) analizzando i sintomi e i dati demografici del paziente.

## Obiettivo del Progetto

L'obiettivo è fornire uno strumento di supporto decisionale rapido ed efficace per la scelta di comuni farmaci da banco. Analizzando 20 diversi parametri clinici e demografici, il sistema mira a ridurre il rischio di somministrazioni non ottimali.

## Funzionalità Chiave

*  **Modello ML Avanzato:** Sfrutta un **Random Forest Classifier** addestrato su un dataset di 1000 record, ottenendo un'accuratezza del **98%** in fase di test.
* **Interfaccia CLI Interattiva:** Un'interfaccia a riga di comando guida l'utente nell'inserimento dei dati, con validazione immediata degli input.
* **Trasparenza nelle Predizioni:** Oltre al farmaco consigliato, il sistema mostra il **livello di confidenza** (%) e le probabilità associate a ciascuna opzione.
* **Predizioni in Batch:** Possibilità di elaborare interi file CSV contenenti dati di più pazienti in un'unica operazione, salvando i risultati per analisi successive.
* **Archiviazione Dati:** I nuovi casi clinici vengono salvati in un file storico (`real_patients.csv`), contribuendo a creare un dataset per futuri miglioramenti del modello.

## Tecnologie

Il progetto è sviluppato in **Python 3.13.7** e utilizza le seguenti librerie:
* **scikit-learn:** Per la creazione del modello Random Forest, le pipeline di elaborazione e il preprocessing dei dati.
* **pandas:** Per la manipolazione dei dati e la gestione dei file CSV.
* **numpy:** Per le operazioni numeriche.

##  Guida Rapida

1.  **Clona il repository:**
    ```bash
    git clone [https://github.com/alespes/Esame-SistemiMultimediali.git](https://github.com/alespes/Esame-SistemiMultimediali.git)
    cd Esame-SistemiMultimediali
    ```
2.  **Installa le dipendenze necessarie:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Esegui l'applicazione:**
    Assicurati che il file del dataset `enhanced_fever_medicine_recommendation.csv` sia presente nella cartella del progetto.
    ```bash
    python raccomandazione_medicina.py
    ```
4.  Segui le istruzioni a schermo per inserire i dati di un nuovo paziente o per analizzare un file CSV esistente.

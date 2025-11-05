"""
SISTEMA COMPLETO DI RACCOMANDAZIONE MEDICINALI
- Training del modello
- Input interattivo pazienti reali
- Batch prediction da CSV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer 
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Variabili globali per il modello
model = None
preprocessor = None
label_encoder = None

# ============================================================================
# PARTE 1: TRAINING DEL MODELLO
# ============================================================================

def train_model(csv_file='enhanced_fever_medicine_recommendation.csv'):
    """Addestra il modello sui dati storici"""
    global model, preprocessor, label_encoder
    
    TARGET_COLUMN = 'Recommended_Medication'
    numerical_features = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Physical_Activity', 'Heart_Rate']
    binary_features = ['Headache', 'Body_Ache', 'Fatigue']
    categorical_features = [
        'Fever_Severity', 'Gender', 'Chronic_Conditions', 'Allergies', 
        'Smoking_History', 'Alcohol_Consumption', 'Diet_Type',
        'Blood_Pressure', 'Previous_Medication'    
    ]
    
    print("="*60)
    print("🏋️ TRAINING DEL MODELLO")
    print("="*60)
    
    # Carica dati
    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' non trovato!")
        return False
    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    
    print(f"✅ Dataset caricato: {len(df)} campioni")
    
    # Pulizia dati numerici
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rimuovi colonne numeriche vuote
    numerical_features_valid = [col for col in numerical_features 
                                if col in df.columns and df[col].notna().any()]
    
    # Pulizia dati binari
    binary_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Si': 1, 'si': 1, 
                      'YES': 1, 'NO': 0, '1': 1, '0': 0, 1: 1, 0: 0, 
                      's' : 1, 'n': 0, 'S': 1, 'N': 0, 'Y': 1, 'N': 0, 'y': 1}
    for col in binary_features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().replace(binary_mapping)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Separazione features e target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🏷️ Classi trovate: {list(label_encoder.classes_)}")
    
    # Pipeline preprocessing
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_valid),
            ('cat', categorical_pipeline, categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    
    X_final = preprocessor.fit_transform(X)
    
    # Training
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced',
        max_depth=20, min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Valutazione
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Modello addestrato!")
    print(f"   Accuratezza: {accuracy:.2%}")
    return True

# ============================================================================
# PARTE 2: FUNZIONE DI PREDIZIONE
# ============================================================================

def predict_medication(patient_data):
    """Predice il medicinale per un paziente"""
    global model, preprocessor, label_encoder
    
    if model is None or preprocessor is None or label_encoder is None:
        raise Exception("Modello non addestrato! Esegui prima train_model()")
    
    # Pulizia dati binari
    binary_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Si': 1, 'si': 1, 
                      'YES': 1, 'NO': 0, '1': 1, '0': 0, 1: 1, 0: 0, 
                      's' : 1, 'n': 0, 'S': 1, 'N': 0, 'Y': 1, 'N': 0, 'y': 1}
    
    # Converti in DataFrame
    df = pd.DataFrame([patient_data])
    
    # Pulizia numerici
    numerical_cols = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Physical_Activity', 'Heart_Rate']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Pulizia binari
    binary_cols = ['Headache', 'Body_Ache', 'Fatigue']
    for col in binary_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().replace(binary_mapping)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Preprocessing e predizione
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    proba = model.predict_proba(X_processed)[0]
    
    medication = label_encoder.inverse_transform(prediction)[0]
    confidence = proba.max() * 100
    
    proba_dict = {label_encoder.classes_[i]: proba[i]*100 
                  for i in range(len(label_encoder.classes_))}
    
    return medication, confidence, proba_dict

# ============================================================================
# PARTE 3: FUNZIONI DI INPUT INTERATTIVO
# ============================================================================

def valida_numero(prompt, min_val=None, max_val=None):
    """Valida input numerico"""
    while True:
        try:
            val = float(input(prompt).strip())
            if min_val is not None and val < min_val:
                print(f"⚠️ Valore minimo: {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"⚠️ Valore massimo: {max_val}")
                continue
            return val
        except ValueError:
            print("❌ Inserisci un numero valido")

def valida_scelta(prompt, opzioni):
    """Valida scelta multipla"""
    print(prompt)
    for i, opt in enumerate(opzioni, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            idx = int(input(f"Scegli (1-{len(opzioni)}): ").strip()) - 1
            if 0 <= idx < len(opzioni):
                return opzioni[idx]
            print(f"❌ Scegli un numero tra 1 e {len(opzioni)}")
        except ValueError:
            print("❌ Inserisci un numero valido")

def valida_si_no(prompt):
    """Valida Si/No"""
    while True:
        resp = input(f"{prompt} (Si/No): ").strip().lower()
        if resp in ['si', 's', 'yes', 'y', 'Si']:
            return 'Yes'
        if resp in ['no', 'n', 'No']:
            return 'No'
        print("❌ Rispondi 'Si' o 'No'")

def collect_patient_data():
    """Raccoglie dati paziente interattivamente"""
    print("\n" + "="*60)
    print("🏥 INSERIMENTO NUOVO PAZIENTE")
    print("="*60 + "\n")
    
    patient = {}
    
    # Anagrafici
    print("📋 DATI ANAGRAFICI")
    print("-"*40)
    patient['Age'] = valida_numero("Età del paziente (0-120): ", 0, 120)
    patient['Gender'] = valida_scelta("Sesso:", ['Male', 'Female', 'Other'])
    patient['BMI'] = valida_numero("BMI (Indice Massa Corporea, 15-50): ", 15, 50)
    
    # Sintomi
    print("\n🌡️ SINTOMI E TEMPERATURA")
    print("-"*40)
    patient['Temperature'] = valida_numero("Temperatura corporea (35°C - 42°C): ", 35, 42)
    
    # Auto-determina severità
    temp = patient['Temperature']
    if temp < 37.5:
        patient['Fever_Severity'] = 'Normal'
    elif temp < 38.5:
        patient['Fever_Severity'] = 'Low'
    elif temp < 39.5:
        patient['Fever_Severity'] = 'Moderate'
    else:
        patient['Fever_Severity'] = 'High'
    print(f"   → Severità febbre: {patient['Fever_Severity']}")
    
    patient['Headache'] = valida_si_no("Il paziente ha mal di testa?")
    patient['Body_Ache'] = valida_si_no("Il paziente ha dolori muscolari?")
    patient['Fatigue'] = valida_si_no("Il paziente è affaticato?")
    patient['Heart_Rate'] = valida_numero("Frequenza cardiaca (bpm 40-200): ", 40, 200)
    
    # Storia clinica
    print("\n📖 STORIA CLINICA")
    print("-"*40)
    patient['Chronic_Conditions'] = valida_scelta(
        "Condizioni croniche:",
        ['None', 'Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Other']
    )
    patient['Allergies'] = valida_scelta(
        "Allergie:",
        ['None', 'Pollen', 'Dust', 'Food', 'Medication', 'Other']
    )
    patient['Blood_Pressure'] = valida_scelta(
        "Pressione sanguigna:",
        ['Normal', 'Low', 'High']
    )
    
    # Stile di vita
    print("\n🏃 STILE DI VITA")
    print("-"*40)
    patient['Smoking_History'] = valida_scelta(
        "Storia di fumo:",
        ['Never', 'Former', 'Current']
    )
    patient['Alcohol_Consumption'] = valida_scelta(
        "Consumo di alcol:",
        ['None', 'Occasional', 'Moderate', 'Heavy']
    )
    patient['Diet_Type'] = valida_scelta(
        "Tipo di dieta:",
        ['Balanced', 'Vegetarian', 'Vegan', 'Mediterranean', 'High-Protein', 'Other']
    )
    patient['Physical_Activity'] = valida_numero(
        "Livello attività fisica (0-10): ", 0, 10
    )
    
    # Ambiente
    print("\n🌍 FATTORI AMBIENTALI")
    print("-"*40)
    patient['Humidity'] = valida_numero("Umidità ambientale (0% / 100%): ", 0, 100)
    patient['AQI'] = valida_numero("Indice qualità aria (AQI 0 - 500): ", 0, 500)
    
    # Farmaci
    print("\n💊 FARMACI")
    print("-"*40)
    patient['Previous_Medication'] = valida_scelta(
        "Farmaci assunti recentemente:",
        ['None', 'Ibuprofen', 'Paracetamol', 'Aspirin', 'Other']
    )
    
    return patient

def show_summary(patient):
    """Mostra riepilogo paziente"""
    print("\n" + "="*60)
    print("📊 RIEPILOGO DATI PAZIENTE")
    print("="*60)
    print(f"\n👤 Dati: {patient['Gender']}, {patient['Age']} anni, BMI {patient['BMI']}")
    print(f"🌡️ Temperatura: {patient['Temperature']}°C ({patient['Fever_Severity']})")
    print(f"💓 Frequenza cardiaca: {patient['Heart_Rate']} bpm")
    print(f"🩺 Pressione: {patient['Blood_Pressure']}")
    print(f"😷 Sintomi: Mal di testa={patient['Headache']}, Dolori={patient['Body_Ache']}, Fatica={patient['Fatigue']}")
    print(f"💊 Farmaco precedente: {patient['Previous_Medication']}")

def save_to_csv(patient, filename='real_patients.csv'):
    """Salva paziente in CSV"""
    df_new = pd.DataFrame([patient])
    df_new['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(filename, index=False)
    print(f"✅ Dati salvati in '{filename}'")

# ============================================================================
# PARTE 4: BATCH PREDICTION
# ============================================================================

def batch_prediction(csv_file):
    """Esegue predizioni su un file CSV di pazienti"""
    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' non trovato!")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n✅ Caricati {len(df)} pazienti da '{csv_file}'")
    
    results = []
    for idx, row in df.iterrows():
        try:
            patient = row.to_dict()
            med, conf, proba = predict_medication(patient)
            results.append({
                'Index': idx,
                'Age': patient.get('Age', 'N/A'),
                'Temperature': patient.get('Temperature', 'N/A'),
                'Recommended': med,
                'Confidence': f"{conf:.1f}%"
            })
        except Exception as e:
            results.append({
                'Index': idx,
                'Age': 'ERROR',
                'Temperature': 'ERROR',
                'Recommended': str(e),
                'Confidence': '0%'
            })
    
    # Mostra risultati
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 RISULTATI BATCH PREDICTION")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Salva risultati
    output_file = csv_file.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Risultati salvati in '{output_file}'")

# ============================================================================
# PARTE 5: MENU PRINCIPALE
# ============================================================================

def main():
    """Funzione principale con menu"""
    print("\n" + "="*60)
    print("🏥 SISTEMA RACCOMANDAZIONE MEDICINALI v2.0")
    print("="*60)
    
    # Training iniziale
    print("\n📚 Caricamento e training del modello...")
    success = train_model()
    
    if not success:
        print("\n❌ Impossibile avviare il sistema.")
        print("   Verifica che il file 'enhanced_fever_medicine_recommendation.csv'")
        print("   sia presente nella directory corrente.")
        return
    
    # Menu principale
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPALE")
        print("="*60)
        print("1. 🆕 Inserisci nuovo paziente (interattivo)")
        print("2. 📁 Batch prediction da CSV")
        print("3. 📊 Visualizza storico pazienti")
        print("4. 🔄 Riallena il modello")
        print("5. ❌ Esci")
        
        choice = input("\nScegli un'opzione (1-5): ").strip()
        
        if choice == '1':
            # Inserimento nuovo paziente
            try:
                patient = collect_patient_data()
                show_summary(patient)
                
                confirm = input("\n✅ Confermi i dati? (Si/No): ").strip().lower()
                if confirm in ['si', 's', 'yes', 'y']:
                    # Predizione
                    med, conf, proba = predict_medication(patient)
                    
                    print("\n" + "="*60)
                    print("💊 RACCOMANDAZIONE MEDICINALE")
                    print("="*60)
                    print(f"\n🎯 Farmaco raccomandato: {med}")
                    print(f"📊 Confidenza: {conf:.1f}%")
                    print("\n📈 Probabilità per ogni classe:")
                    for medication, prob in proba.items():
                        bar = "█" * int(prob/5)
                        print(f"   {medication:15} {prob:5.1f}% {bar}")
                    
                    # Salva
                    save = input("\n💾 Salvare i dati? (Si/No): ").strip().lower()
                    if save in ['si', 's', 'yes', 'y']:
                        patient['Recommended_Medication'] = med
                        patient['Confidence'] = conf
                        save_to_csv(patient)
                else:
                    print("❌ Dati scartati.")
            except Exception as e:
                print(f"\n❌ Errore: {e}")
        
        elif choice == '2':
            # Batch prediction
            file = input("\nNome file CSV (default: real_patients.csv): ").strip()
            if not file:
                file = 'real_patients.csv'
            batch_prediction(file)
        
        elif choice == '3':
            # Visualizza storico
            if os.path.exists('real_patients.csv'):
                df = pd.read_csv('real_patients.csv')
                print(f"\n📊 Storico: {len(df)} pazienti registrati")
                print("\nUltimi 5 pazienti:")
                cols = ['Age', 'Gender', 'Temperature', 'Fever_Severity', 'Recommended_Medication']
                available_cols = [c for c in cols if c in df.columns]
                print(df[available_cols].tail().to_string(index=False))
            else:
                print("\n❌ Nessuno storico trovato in 'real_patients.csv'")
        
        elif choice == '4':
            # Riallena modello
            confirm = input("\n⚠️ Riallena il modello? (Si/No): ").strip().lower()
            if confirm in ['si', 's', 'yes', 'y']:
                train_model()
        
        elif choice == '5':
            # Esci
            print("\n👋 Grazie per aver usato il sistema. Arrivederci!")
            break
        
        else:
            print("\n❌ Scelta non valida. Riprova.")

# ============================================================================
# AVVIO PROGRAMMA
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programma interrotto. Arrivederci!")
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()
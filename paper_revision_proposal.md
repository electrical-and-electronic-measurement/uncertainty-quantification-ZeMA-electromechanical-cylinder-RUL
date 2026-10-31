# Proposta di revisione della bozza del paper — allineamento ai risultati finali del Caso di Studio A

**Bozza esaminata:** `elsarticle-template-num.tex` (*Extending the GUM Framework to Deep Neural Networks…*).
**Risultati di riferimento:** commit `3416f93` — `case_study_A_report.md` §11.1, `case_study_A_report_budget.md`, `case_study_A_dataset_report.md`, `pipeline_run_budget_first100_datasheet.log`, `pipeline_run_budget_first100_datasheet_blocked50.log`.

La bozza riporta il **budget originale** (tutta la *u* dichiarata dal dataset propagata come rumore indipendente per lettura) e una diagnosi della sovracopertura — "il canale pressione è amplificato dalla normalizzazione z-score" — che il repository ha poi superato. La causa reale è che la *u* dichiarata è in gran parte **sistematica**, e una componente sistematica comune a tutti i cicli si cancella nella normalizzazione quando training e inferenza usano lo stesso strumento. Questo documento propone, sezione per sezione, cosa cambiare, perché, e il testo LaTeX sostitutivo.

Configurazione adottata per i numeri principali (raccomandazione di `case_study_A_report_budget.md` §5): ingresso di base (11 canali, primi 100 campioni), modello `datasheet` per la parte casuale, split casuale. I modelli `noise`/`bound` e lo split a blocchi compaiono come controlli. La configurazione senza pressione resta una scelta degli autori (§5.5 sotto).

---

## 0. Numeri: bozza e risultati finali a confronto

| Grandezza | Bozza | Finale (`datasheet`, split casuale) | Fonte |
|---|---|---|---|
| Cicli train / validation / test | 3813 / – / 953 | 3242 / 571 / 953 | log `datasheet` |
| RMSE / R² | 0.064 / 0.951 | 0.0664 / 0.948 | log |
| MAE / mediana \|e\| | 0.046 / 0.033 | 0.047 / 0.034 | CSV `budget_first100_datasheet` |
| u_physical MC (LPU) | 0.984 (0.641) | u_rand 0.076 (0.073) | log |
| LPU vs MC, mediana | 36.5% | 3.6% (`noise` 4.7%, `bound` 6.3%) | log, report budget §3 |
| \|media MC − f(x)\| medio | (0.46, non riportato) | 0.010 | log |
| u_training | 0.049 | 0.051 | log |
| termine residuo da validation | – | 0 (clippato) | log |
| u_c (stesso strumento) | 0.985 | **0.093** | log |
| Copertura k=2 / k=1 | 100% / – | 99.6% / 89.2% | log, README |
| u_c / MAE; corr(u_c, \|e\|) | 14–21×; 0.26 | 1.97; 0.45 | log |
| rms u_rand propagata / consentita dai residui | – | 0.078 / 0.032 | report budget §3.1 |
| u_sys MC (LPU), trasferimento | – | 0.782 (0.598) | log |
| u_c,transfer | – | 0.788 | log |
| Tipo di u_physical | A | **B** | report §5, dataset report F3 |
| Split a blocchi (50 cicli) | – | RMSE 0.074, R² 0.95, LPU/MC 3.8%, u_c 0.093, copertura k=2 99.3% | log `datasheet_blocked50` |

---

## 1. Frontmatter: abstract, highlights, keywords

**Cosa cambiare.** L'abstract non menziona il caso di studio; gli highlights sono ancora placeholder.

**Perché.** Il caso di studio è l'unica validazione empirica del metodo, e il suo risultato (la Fase 5 rileva un budget errato e ne guida la correzione) è il messaggio più concreto del paper.

**Testo proposto** — da aggiungere alla fine dell'abstract, prima di "The proposed methodology provides…" o in coda:

```latex
Applied to remaining-useful-life estimation on the ZeMA electromechanical
cylinder data set, the validation phase of the methodology exposed an
uninformative budget ($u_c \approx 0.98$ on a $[0,1]$ measurand, 36.5\%
disagreement between analytical and Monte Carlo propagation) caused by
propagating a mostly systematic declared input uncertainty as independent
noise. Separating random and systematic components reduced the combined
standard uncertainty to $0.093$ and the disagreement to 3.6\%.
```

**Highlights proposti:**

```latex
\begin{highlights}
\item A five-phase GUM methodology for uncertainty in neural-network measurements.
\item Criteria select analytical propagation or Monte Carlo simulation per network.
\item Classifier outputs are treated as nominal properties with PMF dispersion.
\item Validation exposed a systematic input component propagated as random noise.
\item Separating random and systematic input parts made the budget informative.
\end{highlights}
```

(Ogni highlight rispetta il limite Elsevier di 85 caratteri, spazi inclusi.)

---

## 2. Sezione 1 — Introduzione e §1.1 Scope and contributions

**Cosa cambiare.**
1. Nel paragrafo rosso su Angrisani et al. la frase "Integrates measurement-model calibration…" è senza soggetto.
2. In §1.1 aggiungere una frase sul risultato del caso di studio.

**Testo proposto** (1), sostituisce le ultime due frasi del paragrafo rosso:

```latex
The present work builds on this calibration-oriented perspective but addresses
a broader methodological scope: it integrates measurement-model calibration
with analytical and Monte Carlo propagation of input uncertainty, an explicit
treatment of training- and model-related contributions, an extension to
nominal classification results, and empirical validation and reporting within
a unified five-phase framework.
```

**Testo proposto** (2), sostituisce "Section~\ref{sec:case_study} applies the framework in a real measurement setting":

```latex
Section~\ref{sec:case_study} applies the framework to a real measurement
setting, where the validation phase detects and diagnoses an incorrect
interpretation of the input uncertainties declared with the data set,
```

---

## 3. Sezione 2 — LPU per reti neurali (§2.1 Figura 1, §2.2 Linearity)

### 3.1 Figura 1 (neurone e MLP)

**Cosa cambiare.** Sostituire l'immagine presa da Carratù et al. (`NN_neuron_with_bias.png`, nota verde "LICENZA OPEN, MA MEGLIO RIFARE") con `paper_figures/nn_neuron_lpu.pdf` di questo repository. La nuova figura è disegnata in TikZ con la notazione della Sezione 2:

- $\sigma_i$ è la funzione di attivazione, come chiede la nota verde;
- $\mathbf{W}_i$ e $\mathbf{b}_i$ sono pesi e bias del layer $i$ (elementi $W_{i,jk}$ e $b_{i,j}$);
- $\mathbf{x}^{(i)}$ è l'uscita del layer $i$, con $\mathbf{x}^{(0)} = \mathbf{x}$ e $\mathbf{x}^{(L)} = \mathbf{y}$;
- $\mathbf{H}_i$ è definita come nell'Eq.~(H_i_definition).

Il pannello (a) mostra il neurone $j$ del layer $i$, con la preattivazione $z_{i,j}$ e il coefficiente di sensibilità $\sigma_i'(z_{i,j})\,W_{i,jk}$. Il pannello (b) mostra l'MLP con lo stesso neurone evidenziato e, sotto, la propagazione layer per layer della matrice di covarianza dell'Eq.~(Law_of_Propagation_of_Uncertainty_for_NN). In dimensione naturale la figura misura 18,7 × 8,4 cm: va inclusa con `width=\linewidth`. Compila con `pdflatex` semplice.

**Coerenza dei simboli** (nota verde "attenzione alla definizione dei simboli sotto"):
- il testo che precede l'equazione scrive $\mathbf{H}(i)$ e $\mathbf{W}(i)$, mentre le equazioni usano $\mathbf{H}_i$ e $\mathbf{W}_i$: uniformare il testo alla forma con pedice;
- definire $L$ come numero di layer e $\mathbf{x}^{(0)} = \mathbf{x}$, $\mathbf{x}^{(L)} = \mathbf{y}$, che la figura usa.

**Testo proposto**:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{nn_neuron_lpu.pdf}
    \caption{(a) Artificial neuron $j$ of layer $i$: the outputs
    $x^{(i-1)}_k$ of the previous layer are weighted by $W_{i,jk}$ and summed
    with the bias $b_{i,j}$ into the preactivation $z_{i,j}$, which the
    activation function $\sigma_i$ maps to the output $x^{(i)}_j$; the
    sensitivity coefficient of $x^{(i)}_j$ with respect to $x^{(i-1)}_k$ is
    $\sigma_i'(z_{i,j})\,W_{i,jk}$. (b) Multilayer perceptron with input
    $\mathbf{x}^{(0)} = \mathbf{x}$ and output $\mathbf{x}^{(L)} = \mathbf{y}$;
    neuron $j$ of panel (a) and its incoming weights are highlighted. The
    covariance matrix of the input estimates is propagated layer by layer,
    each layer contributing the sensitivity matrix $\mathbf{H}_i\mathbf{W}_i$
    (Eqs.~\eqref{eq:Law_of_Propagation_of_Uncertainty_for_NN}
    and~\eqref{eq:H_i_definition}), so that $\mathbf{U}_{\mathbf{y}} =
    \mathbf{C}\mathbf{U}_{\mathbf{x}}\mathbf{C}^{\top}$ with
    $\mathbf{C} = \mathbf{H}_L\mathbf{W}_L\cdots\mathbf{H}_1\mathbf{W}_1$
    (Eq.~\eqref{eq:measurement_covariance_matrix}).}
    \label{fig:Schematic_representation_artificial_neuron}
\end{figure}
```

### 3.2 §2.2 Linearity

**Cosa cambiare.** Nessuna modifica di sostanza. Facoltativo: aggiungere una frase alla fine dell'ultimo paragrafo (dopo il risultato di Lin et al.).

**Perché.** Nel caso di studio il disaccordo LPU/MC passa da 36.5% a 3.6% sulla **stessa rete**, cambiando solo il modello dell'incertezza in ingresso. L'adeguatezza della linearizzazione dipende quindi dall'ampiezza delle perturbazioni rispetto al dominio di training, non solo dalla profondità della rete.

```latex
The adequacy of the linearization depends on the magnitude of the input
perturbations relative to the training domain as much as on the network
itself: in the case study of Section~\ref{sec:case_study}, the same network
gives a 36.5\% or a 3.6\% LPU--Monte Carlo disagreement depending only on how
the declared input uncertainty is interpreted.
```

---

## 4. Sezione 3 — Proprietà nominali

Nessuna modifica: il caso di studio è di regressione e non tocca questa sezione.

---

## 5. Sezione 4 — Metodologia (modifiche di coerenza suggerite dal caso di studio)

Il caso di studio ha fatto emergere tre elementi che la metodologia non prevede ancora in modo esplicito: la scomposizione casuale/sistematica dell'incertezza in ingresso, lo scenario d'uso (stesso strumento o trasferimento) e alcuni controlli di validazione. Senza queste aggiunte, la Sezione 5 applicherebbe passi che la Sezione 4 non descrive.

### 5.1 Phase 2 — *Data Uncertainty Sources*

**Cosa cambiare.** Aggiungere due paragrafi dopo quello su *Input data uncertainty*. Il ragionamento va reso esplicito con i riferimenti puntuali alla GUM e al VIM (clausole verificate sulle edizioni ufficiali BIPM di JCGM 100:2008 e JCGM 200:2012). Le chiavi `JCGM100`, `JCGM102` e `JCGM200` sono da aggiungere al `.bib` del paper accanto a `JCGM101`, che la bozza usa già; le voci BibTeX sono in `case_study_A_report.md` §11.1. Nel `.bib` della tesi (`JCGM-documents.bib`) le stesse voci si chiamano `JCGMGUM-guide-to-uncertainty-in-measurement`, `JCGM102-GUM-supplement2` e `JCGMVIM3-International-vocabulary-of-metrology`.

```latex
The declared standard uncertainty $u(x_i)$ of an input does not, in general,
say how the underlying errors are correlated between observations. In the
terms of the VIM, the error of an observation combines a random component,
which varies unpredictably between replicate observations, and a systematic
component, which remains constant over them \cite[2.19, 2.17]{JCGM200}:
$u^2 = u_{\mathrm{rand}}^2 + u_{\mathrm{sys}}^2$. The systematic component
is common to all observations made with the same instrument, which makes
these observations correlated \cite[5.2.4]{JCGM100}; equivalently, it can
be introduced as an additional input quantity shared by all of them
\cite[F.1.2.3, F.1.2.4]{JCGM100}. Such correlations cannot be ignored when
they are significant \cite[5.2.5]{JCGM100}. The split concerns the
correlation structure, not the method of evaluation: the Type~A/B
classification ``is not meant to indicate that there is any difference in
the nature of the components'' \cite[3.3.4]{JCGM100} and is not a
substitute for the words random and systematic \cite[3.3.3]{JCGM100}.
Whether a component acts as random or systematic depends on the measurement
model in which the quantity is used \cite[E.3.6]{JCGM100}, and for a
learned model this includes the preprocessing fitted on the training data.

When the same instrument is used for training and inference, a systematic
error common to all observations of an input enters the training data and
the inference data in the same way. An offset or gain error, in particular,
is an affine map of that input, which a normalization fitted on the training
data removes exactly: the sensitivity of the prediction to it is zero, and
only the random component contributes to the uncertainty of the prediction.
The systematic component becomes relevant when the model is deployed with a
different instrument, where it is propagated as a shift fully correlated
between the observations of an input (correlation coefficient $+1$,
\cite[5.2.2, Note~1]{JCGM100}). If the data source does not separate the
two components, the data constrain them: only the random component can
contribute to the observed variability between observations, so it is
compared with that variability as a separate group \cite[3.4.3]{JCGM100},
and the cross-observation standard deviation $s$ of an input gives the
upper bound $u_{\mathrm{rand}} \le s$, since
$s^2 = \mathrm{var}(\text{true}) + u_{\mathrm{rand}}^2$. A declared $u$
larger than $s$ shows that a model of independent observations is
incomplete and must be revised \cite[3.4.2]{JCGM100}: most of $u$ is then
systematic. This treatment assumes that the systematic component is
constant over the data set; a drift correlated with the measurand would not
cancel and would be confounded with it.
```

### 5.2 Phase 4 — Eq. (combined) e notazione

**Cosa cambiare.**
1. Chiarire che u_physical dipende dallo scenario d'uso.
2. Precisare come si stima u_calibration quando si dispone di valori di riferimento su un validation set: è il termine che nel repository si chiama `u_model`, e la guida (`case_study_A_guide.md:121`) lo chiama già `u_calibration`.

**Testo proposto**, da aggiungere dopo il paragrafo rosso che segue le Eq. (combined)/(model_uncertainty_decomposition):

```latex
The physical-quantity term depends on the intended use. For use with the same
instrument as in training, $u_{\mathrm{physical}}$ is obtained by propagating
the random input components only; for deployment with another instrument,
the systematic components are propagated in addition, as a shift common to
all observations of an input, and the two results are reported separately.
When reference values of the measurand are available on a validation set
not used for training, the calibration term can be estimated from the
residuals $e$ without counting the propagated variances twice,
\begin{equation}\label{eq:u_calibration_validation}
u_{\mathrm{calibration}}^2 = \overline{e^2} - \overline{u_{\mathrm{training}}^2}
- \overline{u_{\mathrm{physical}}^2},
\end{equation}
clipped at zero; a negative value indicates that the propagated contributions
already exceed the observed residual variance.
```

### 5.3 Phase 5 — *Validation Procedures*

**Cosa cambiare.** Dopo il paragrafo su "Empirical validation on independent test data…", aggiungere i controlli che nel caso di studio si sono rivelati decisivi.

```latex
Over-coverage is as much a defect as under-coverage: intervals that are too
wide carry little information about which predictions to trust. Coverage
should therefore be checked at more than one coverage factor (e.g., $k=1$ and
$k=2$), and the ratio between the mean combined uncertainty and the mean
absolute error should be reported. Three further checks help to locate the
cause of a discrepancy: (i) the residuals bound the propagated random input
uncertainty, $\overline{u_{\mathrm{physical}}^2} \le \overline{e^2} -
\overline{u_{\mathrm{training}}^2}$; (ii) a large difference between the
Monte Carlo mean and the network output $f(\mathbf{x})$ indicates that the
input perturbations leave the training domain; (iii) for data indexed in time
or cycle number, holding out whole blocks of consecutive observations shows
whether the test error is inflated by near-duplicates of training samples.
```

E nel paragrafo su LPU vs Monte Carlo, dopo "…observed discrepancy from the Monte Carlo benchmark.", aggiungere:

```latex
A large discrepancy may also indicate an incorrect input-uncertainty model
rather than an intrinsically nonlinear network; the input model should be
reviewed (Phase~2) before Monte Carlo is adopted as primary.
```

### 5.4 Phase 5 — *Reporting Requirements*

Aggiungere alla fine:

```latex
When the random and systematic input components are not separated by the
data source, the report should state how each was estimated, which inputs
rely on a manufacturer specification and which on a data-based bound, and
the stationarity assumption under which the systematic component cancels.
```

### 5.5 Figura 2 (decision tree)

**Cosa cambiare.** Sostituire `decision_tree.tex`/`decision_tree.pdf` con la versione in `paper_figures/` di questo repository. La figura attuale è marcata "SISTEMARE IL DIAGRAMMA". In dimensione naturale è alta circa 50 cm, quindi con `width=0.6\linewidth` diventa illeggibile; inoltre usa ancora la notazione $u_A$/$u_B$ e non compila senza la classe `standalone`. La nuova versione:

- è impaginata su **due colonne**: Fasi 1–3 a sinistra, Fasi 4–5 a destra. In dimensione naturale misura 17,5 × 16,7 cm, quindi sta su una pagina con `width=\linewidth`;
- **Fase 1**: dichiara l'uso previsto (stesso strumento del training o un altro strumento);
- **Fase 2**: nuovo nodo "la correlazione della *u* d'ingresso è dichiarata?". Se no, si confronta *u* con la dispersione *s*, si scompone *u*² = *u*_rand² + *u*_sys² con *u*_rand ≤ *s* e si citano GUM 5.2.4 e F.1.2.4;
- **Fase 4**: usa le Eq. (combined)/(model_uncertainty_decomposition) del paper invece di $u_A$/$u_B$, più un nodo "uso previsto": stesso strumento → solo *u*_rand; altro strumento → anche *u*_sys, riportati separatamente;
- **Fase 5**: aggiunge il test indipendente (hold-out a blocchi per dati indicizzati nel tempo o nel ciclo) e il controllo dei residui. Il disaccordo LPU/MC passa prima per il nodo "modello d'ingresso coerente?". La copertura si verifica a *k* = 1 e *k* = 2, in entrambe le direzioni;
- **ritorni di validazione**: tre connettori con lettera (A → Fase 3, B e C → Fase 2) al posto delle lunghe linee tratteggiate;
- compila con `pdflatex` semplice: il ritaglio della pagina è nel file, non serve `standalone`.

I numeri di sezione nella figura (Sec.~4.1–4.5, Sec.~3) e l'etichetta "Eq.~LPU-for-NN" sono testo fisso, come nella versione attuale: vanno aggiornati se cambia la numerazione.

**Testo proposto** (ambiente `figure` della Sezione 4):

```latex
\begin{figure}[p]
    \centering
    \includegraphics[width=\linewidth]{decision_tree.pdf}
    \caption{Decision-tree overview of the five-phase methodology. Phase~1
    branches the task into a regression track (blue) or a classification
    track (orange, Section~\ref{sec:extending_GUM_to_Nominal_Properties}) and
    declares the intended use of the model. Phase~2 checks whether the
    correlation structure of the declared input uncertainty is stated; if not,
    the declared $u$ is compared with the spread $s$ between observations and
    split into a random and a systematic component
    (Section~\ref{sec:Uncertainty_sources_ML}). Phase~3 applies the three
    propagation-method criteria (model accessibility, architectural and
    differentiability compatibility, computational constraints). Phase~4
    combines the contributions of Eqs.~\eqref{eq:combined_std_uncertainty}
    and~\eqref{eq:model_uncertainty_decomposition}, propagating the systematic
    input component only when the model is deployed with a different
    instrument. Phase~5 validates on independent data; its returns (dashed,
    connectors A--C) lead to Phase~3 when the linearity assumption is violated
    and the input model is consistent (A), and to Phase~2 when the input model
    is inconsistent (B) or the empirical coverage departs from the nominal
    value in either direction (C).}
    \label{fig:decision_tree}
\end{figure}
```


### 5.6 Figura 3 (scomposizione dell'incertezza)

**Cosa cambiare.** Sostituire `measurement_uncertainty_decomposition.png` (nota verde "RIFARE IMMAGINE") con `paper_figures/uncertainty_decomposition.pdf` di questo repository. L'albero ha due macro-componenti, come nel paragrafo rosso che precede la figura (Angrisani et al.). Ogni ramo riporta nell'intestazione l'equazione corrispondente del paper:
- incertezza della grandezza fisica: $\mathbf{U}_{\mathbf{y}} = \mathbf{C}\mathbf{U}_{\mathbf{x}}\mathbf{C}^{\top}$ (Eq. measurement_covariance_matrix);
- incertezza del modello di misura: Eq. (model_uncertainty_decomposition).

Ogni contributo è collegato al proprio termine:
- dati di training finiti e non rappresentativi: non hanno un termine proprio e agiscono attraverso gli altri;
- variabilità del processo di training: $u_{\mathrm{training}}$, da un ensemble di seed (Type A);
- selezione del modello e dell'architettura: $u_{\mathrm{architecture}}$;
- effetti di approssimazione e generalizzazione: $u_{\mathrm{calibration}}$, da valori di riferimento e dal validation set.

Nel ramo fisico, le correlazioni d'ingresso richiamano le componenti casuale e sistematica di Fase 2. In dimensione naturale la figura misura 14,2 × 8,4 cm: `width=0.9\linewidth` va bene. Compila con `pdflatex` semplice.

Il paragrafo rosso dice che l'incertezza del modello è scomposta in "training process, selection and architecture, approximation and validation"; la figura usa la stessa suddivisione. Va invece allineata l'Eq. (combined_std_uncertainty) commentata nella bozza, che usa ancora $u_{\mathrm{generalization}}$: va eliminata, oppure rinominata $u_{\mathrm{calibration}}$.

**Testo proposto**:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{uncertainty_decomposition.pdf}
    \caption{Components of the measurement uncertainty of ML-based
    measurement systems. The physical-quantity uncertainty
    $u_{\mathrm{physical}}$ arises from the input quantities, including their
    correlations, and is propagated through the learned measurement function
    (Eq.~\eqref{eq:measurement_covariance_matrix}); the measurement-model
    uncertainty $u_{\mathrm{model}}$ arises from inferring that function from
    finite data and is decomposed as in
    Eq.~\eqref{eq:model_uncertainty_decomposition}. Finite and
    non-representative training data has no term of its own: it acts through
    the training-process variability and through the approximation and
    generalization effects that calibration against reference values
    characterizes.}
    \label{fig:measurement_uncertainty_decomposition}
\end{figure}
```

---

## 6. Sezione 5 — Caso di studio (riscrittura)

### 6.1 Introduzione della sezione

**Cosa cambiare.**
- "*metrologically evaluated standard uncertainties*" non è corretto. Dorst et al. (2022) non descrivono come sono state ottenute le incertezze ZeMA DAQ; nel file sono costanti per canale oppure pari esattamente all'1%/1.5% della lettura (dataset report F3). Sono quindi valori dichiarati, di **Tipo B**. Questo risolve anche la nota rossa ("depends on the evaluation procedure documented in…").
- Nota rossa su u_calibration: ora il termine **è stimato**, sul validation set (Eq. proposta in §5.2), e risulta 0. Resta escluso solo u_architecture.
- Correggere "Equation \ref" mancante ("of \ref{eq:combined_std_uncertainty}" → "of Eq.~\eqref{…}").

**Testo proposto** — sostituisce il primo paragrafo, la frase "eleven co-located sensors…metrologically evaluated…" e i due paragrafi rossi:

```latex
Few publicly available regression data sets provide a per-sample standard
uncertainty for each input sensor, a prerequisite for an uncertainty budget
built from the inputs rather than estimated post hoc from residuals. The ZeMA
testbed data set of Dorst et al.~\cite{Dorst2021_ZeMA_dataset} is an exception:
eleven sensors recorded synchronized measurements, each annotated with a
standard uncertainty, over 4766 working cycles of a Festo ESBF
electromechanical cylinder run to failure. It is distributed under CC-BY-4.0
with the open-source package \textit{zema\_emc\_annotated}~\cite{ludwig2022_zema_emc_annotated},
which exposes sensor values and standard uncertainties $u(x_i)$ through a
documented Python API.

[paragrafo sul misurando invariato]

The term $u_{\mathrm{physical}}^2(y_j)$ of Eq.~\eqref{eq:combined_std_uncertainty}
is obtained by propagating the per-sample standard uncertainties $u(x_i)$
distributed with the data set. Their evaluation is not described
in~\cite{Dorst2022}; in the file they are either a constant per channel or a
fixed percentage (1\% or 1.5\%) of the reading, consistent with
specification-based values. They are therefore treated as Type B
evaluations, and propagating them does not change their type. Because the
network is white-box and a continuously differentiable activation (softplus)
is used, both the analytical LPU and Monte Carlo propagation are applicable,
so the linearity assumption of Section~\ref{sec:linearity_differentiability}
can be tested directly rather than assumed. Of the model-related terms of
Eq.~\eqref{eq:model_uncertainty_decomposition}, $u_{\mathrm{training}}$ is
evaluated from a random-seed ensemble, a Type A evaluation of an epistemic
contribution, and $u_{\mathrm{calibration}}$ from the residuals on a
validation set (Eq.~\eqref{eq:u_calibration_validation});
$u_{\mathrm{architecture}}$ is excluded because the architecture is kept
fixed.
```

### 6.2 §5.1 Experimental setup

**Cosa cambiare.**
- "11 × 100 readings per cycle" va precisato: sono i **primi 100 campioni (primi 50 ms)** della finestra di 1 s della corsa di ritorno a 2 kHz, perché `ZeMASamples` prende i primi *d* campioni e non *d* campioni distribuiti sulla finestra (dataset report §7).
- Gli accelerometri sono **tre sensori distinti** (plain bearing, piston rod, ball bearing), non gli assi x/y/z di un solo sensore.
- Split: training 3242, validation 571, test 953 (non "80% training, 953 for validation": i 953 cicli sono il test set).
- Aggiungere il modello dell'incertezza in ingresso, con i riferimenti GUM/VIM: qui la versione per il caso di studio, che rimanda alla Sezione 4 per il ragionamento generale (§5.1 di questo documento). `case_study_A_report.md` §11.1 ne contiene la versione estesa e autonoma, con la tabella delle clausole citate.
- Correggere "softplus was trained with double precision in 80%…".

**Testo proposto** (sostituisce l'intera §5.1):

```latex
The study uses \textit{zema\_emc\_annotated} v0.7.1 with input dimension
$11 \times 100 = 1100$: the first 100 samples (50\,ms) of the 1\,s
return-stroke window, recorded at 2\,kHz, for each of the 11 channels (three
accelerometers at the plain bearing, piston rod and ball bearing; active
current; axial force; three motor-current phases; pneumatic pressure; sound
pressure; velocity). An MLP with hidden layers $(128, 64, 32)$ and softplus
activation was trained in double precision (Adam, learning rate $10^{-3}$,
200 epochs, MSE loss) on 3242 cycles; 571 cycles form a validation set and
953 cycles a held-out test set (random split). Inputs are $z$-score
normalized with training-set statistics; normalization is a linear stage of
the measurement model, so GUM propagation through it is exact,
$u(x') = u(x)/\sigma_{\text{train}}$. The point estimate is the output
$f(\mathbf{x})$ of the reference ensemble member.

\textit{Input-uncertainty model.} The data set declares a single
standard uncertainty $u$ per reading, evaluated as Type~B, without stating
its correlation between readings, cycles and channels. Independent readings
would require the cross-cycle standard deviation $s$ of each reading to be
at least $u$; for the pressure, velocity and plain-bearing acceleration
channels $u/s$ is instead $8.8$, $4.3$ and $2.1$ (first 100 samples,
training cycles), so the independence model used in a first evaluation is
falsified by the data \cite[3.4.2]{JCGM100}. Following
Section~\ref{sec:Uncertainty_sources_ML}, reading $i$ of channel $k$ in
cycle $c$ is modelled as
$x_{c,ki} = \xi_{c,ki} + \varepsilon_{c,ki} + \delta_{ki}$, with a random
error $\varepsilon$ that varies between cycles and a systematic error
$\delta$ common to all cycles \cite[2.19, 2.17]{JCGM200}, introduced as an
additional input quantity per reading, independent between channels
\cite[5.2.4, F.1.2.4]{JCGM100}. Since only $\varepsilon$ contributes to the
cross-cycle spread \cite[3.4.3]{JCGM100}, $u_{\mathrm{rand}} \le s$, and
at least $98.7\%$, $93.9\%$ and $74\%$ of $u^2$ is common to all cycles for
the three channels. We estimate
$u_{\mathrm{rand}} = \min(u, s, \sigma_k, \sigma_{\mathrm{ds},k})$, where
$s$ and the white-noise floor $\sigma_k$ (robust estimate from second
differences of adjacent samples) are Type~A upper bounds and
$\sigma_{\mathrm{ds},k}$ is the noise floor published by the manufacturer,
available only for the two Kistler 8704B50M1 accelerometers ($0.004\,g$
rms, Type~B). Because the $z$-score normalization is fitted on the training
cycles of the same instrument, $\delta$ cancels and only the random part is
propagated, $\mathbf{U}_x = \mathrm{diag}(u_{\mathrm{rand}}^2)$. For
transfer to another instrument, $\delta$ is propagated in addition, with
correlation coefficient $+1$ within a channel \cite[5.2.2, Note~1]{JCGM100}
and independently between channels, and reported separately.

Following Phase~4, $\mathbf{U}_x$ is propagated layer by layer through the
trained MLP (Eq.~\eqref{eq:Law_of_Propagation_of_Uncertainty_for_NN}) and
independently by Monte Carlo simulation per GUM Supplement~1~\cite{JCGM101}
(with correlated inputs for the transfer case~\cite{JCGM102}),
with Gaussian input distributions and $M = 10^4$ samples per test cycle. The
training-process contribution $u_{\mathrm{training}}$ is the standard
deviation of the predictions of a 10-seed ensemble (optimization variability
only). The combined standard uncertainty for same-instrument use is
$u_c^2 = u_{\mathrm{physical}}^2 + u_{\mathrm{training}}^2 +
u_{\mathrm{calibration}}^2$, with the Monte Carlo value of
$u_{\mathrm{physical}}$.
```

### 6.3 §5.2 Validation results

**Cosa cambiare.** Sostituire **tutta** la sottosezione. I paragrafi su "Linearity check 36.5%" e "Coverage check 100%", e il lungo paragrafo sull'amplificazione della pressione tramite z-score, descrivono una diagnosi superata (report, status update punto 2). Dell'ultimo paragrafo va conservato solo il punto ancora valido: la normalizzazione fa parte del modello di misura, ed è proprio lei a cancellare la parte sistematica.

**Convergenza MC — azione aperta.** Le variazioni 1.75% / 0.08% (M = 10⁵, 10⁶) sono state misurate sul **budget originale** (u ≈ 0.93 su un ciclo di test). Sul budget corretto non esiste un controllo equivalente. Due opzioni:
(a) dichiarare che il controllo è stato fatto sul budget originale, dove le perturbazioni sono molto più grandi;
(b) **raccomandata:** prima della sottomissione, rieseguire il controllo di convergenza con `case_study_A_analysis_budget.py --random-model datasheet`.
Il testo sotto usa (a), con un segnaposto.

**Testo proposto:**

```latex
Validation follows Phase~5: the analytical estimate is compared with Monte
Carlo propagation, and empirical coverage is assessed on the 953 held-out
cycles (Figure~\ref{fig:zema_lpu_vs_mc}).

\textit{A first budget and its diagnosis.} Propagating the full declared
uncertainty as independent per-reading noise gave a mean
$u_{\mathrm{physical}}^{\mathrm{MC}} = 0.98$, a $36.5\%$ median LPU--Monte
Carlo disagreement, and $100\%$ coverage at $k=2$ with intervals spanning the
whole $[0,1]$ range. The Phase~5 checks located the cause in the input model,
not in the network: perturbations of up to nine training-set standard
deviations took the network far outside its training domain (mean shift of
the Monte Carlo mean from $f(\mathbf{x})$: $0.46$), and the declared
uncertainty of the dominant channels exceeded their cross-cycle spread. This
led to the random/systematic interpretation of Section~\ref{sec:case_study_setup}.

\textit{Linearity check.} With the random input part, the analytical and
Monte Carlo estimates agreed to within $3.6\%$ (median relative difference;
$4.7\%$ and $6.3\%$ with the purely data-based bounds
$u_{\mathrm{rand}} = \min(u, s, \sigma_k)$ and $\min(u, s)$), and the Monte
Carlo mean differs from $f(\mathbf{x})$ by $0.010$ on average. The
first-order approximation is therefore adequate at this input-uncertainty
magnitude; the Monte Carlo value is used in the combination. Increasing $M$
to $10^5$ and $10^6$ changed $u_{\mathrm{physical}}^{\mathrm{MC}}$ by
\textbf{[TBD]} \% and \textbf{[TBD]} \%.

\textit{Coverage check.} The empirical coverage of the $k=2$ expanded interval
was $99.6\%$ ($89.2\%$ at $k=1$) against the $\approx 95\%$ target, with
$\bar u_c / \overline{|e|} = 2.0$. The over-coverage comes from the random
input part: the test residuals allow at most $0.03$ of propagated random input
uncertainty, against $0.08$ computed, so $u_{\mathrm{calibration}}$ estimated
on the validation set (Eq.~\eqref{eq:u_calibration_validation}) is zero. The
excess comes from the velocity and plain-bearing acceleration channels,
whose random part is capped at $s$ because no noise specification is
available for them; a tighter estimate requires repeated static measurements
or a manufacturer noise specification. Fitting $u_{\mathrm{rand}}$ to the
residuals would turn the coverage check into a calibration, and is not done.

\textit{Split check.} Consecutive cycles are nearly identical, and the random
split leaves a neighbouring cycle of most test cycles in the training set.
Holding out whole blocks of 50 consecutive cycles, so that no test cycle is
adjacent to a training cycle, gives RMSE $0.074$ (vs.\ $0.066$),
$R^2 = 0.95$, a $3.8\%$ LPU--Monte Carlo difference, $u_c = 0.093$ and
coverage $99.3\%$: neither the accuracy nor the budget is an artefact of the
random split.
```

Se si preferisce l'opzione (a), sostituire l'ultima frase del paragrafo *Linearity check* con:

```latex
Monte Carlo convergence was verified on the first budget, where the
perturbations are an order of magnitude larger: increasing $M$ to $10^5$ and
$10^6$ changed $u_{\mathrm{physical}}^{\mathrm{MC}}$ by $1.75\%$ and $0.08\%$.
```

### 6.4 Figura `fig:zema_lpu_vs_mc`

**Cosa cambiare.** Sostituire il file con `zema_lpu_vs_mc_budget_first100_datasheet.pdf`, che ha la stessa struttura a due pannelli, generata da `make_validation_figure.py` senza modifiche. Opzione: una figura a due righe, budget originale (`zema_lpu_vs_mc.pdf`) sopra e corretto sotto, per mostrare visivamente la diagnosi.

```latex
\includegraphics[width=0.9\linewidth]{zema_lpu_vs_mc_budget_first100_datasheet.pdf}
\caption{Phase~5 validation on the 953 held-out cycles, random input part
only: (a) analytical LPU vs.\ Monte Carlo estimates of
$u_{\mathrm{physical}}$ per test cycle against the identity line ($3.6\%$
median relative difference); (b) predictions with the $k=2$ expanded
uncertainty band vs.\ ground truth (empirical coverage $99.6\%$).}
```

Nota: il pannello (b) della figura è generato dalla colonna `u_c` del CSV `budget_*`, che è già la combinazione MC corretta (report budget §2.2).

### 6.5 §5.3 Tabella del budget

**Cosa cambiare.** Sostituire con la tabella rivista. Nella tabella attuale vanno corretti anche il tipo di u_physical (A → B) e la virgola decimale in "0,985".

```latex
\begin{table}[t]
\centering
\caption{Uncertainty budget for RUL estimation on the ZeMA cylinder data set
(mean over the 953 held-out cycles; same instrument for training and
inference unless stated). Combined values are means of the per-cycle $u_c$,
so they differ slightly from the quadrature sum of the column means.}
\label{tab:zema_uncertainty_budget}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccl}
\toprule
\textbf{Uncertainty source} & \textbf{Type} & \textbf{$u_i$} & \textbf{Notes} \\
\midrule
$u_{\mathrm{physical}}$. Random part of sensor input (11 channels) & B & $0.076$ & Monte Carlo; datasheet value or Type A data-based cap \\
$u_{\mathrm{training}}$. Training process variability (10-seed ensemble) & A & $0.051$ & optimization variance only \\
$u_{\mathrm{calibration}}$. Residual model error (validation set) & A & $0$ & clipped, Eq.~\eqref{eq:u_calibration_validation} \\
$u_{\mathrm{architecture}}$. Architecture selection & \textemdash & excluded & held fixed $(128,64,32)$, softplus \\
Label (RUL definition) & \textemdash & negligible & deterministic, $y_k = k/N$ \\
\midrule
\textbf{Combined standard uncertainty $u_c(\hat{y})$} & \textemdash & $\mathbf{0.093}$ & coverage $99.6\%$ at $k=2$, $89.2\%$ at $k=1$ \\
\midrule
$u_{\mathrm{sys}}$. Systematic part of sensor input (transfer only) & B & $0.782$ & Monte Carlo; mostly pressure \\
Combined, transfer to another instrument & \textemdash & $0.788$ & \\
\bottomrule
\end{tabular}%
}
\end{table}
```

**Riga "Generalization / distribution shift"**: la tabella attuale la porta con tipo B e "assessed via coverage". Si propone di toglierla dalla tabella: è coperta da u_calibration e dalla verifica sullo split a blocchi. Se la si vuole tenere, va scritta senza tipo e senza valore ("assessed by blocked split, Sec.~\ref{sec:case_study_validation}").

**Testo dopo la tabella** (facoltativo, sul trasferimento e sul limite della LPU):

```latex
The systematic part is relevant only when the model is used with a different,
recalibrated instrument without re-normalization. For this contribution the
LPU and Monte Carlo estimates differ markedly ($0.598$ vs.\ $0.782$), since a
shift common to all readings of a channel is a large perturbation; Monte
Carlo is reported.
```

### 6.6 §5.4 Practical significance

**Cosa cambiare.** Riscrivere. Il messaggio "only the interval is compromised… pressure-channel amplification" è superato. Il nuovo messaggio: la Fase 5 ha esposto due difetti successivi e ne ha indicato la causa; dopo la correzione l'intervallo è informativo ma ancora circa 2 volte troppo largo, per una limitazione del dataset.

```latex
The case study illustrates two failure modes that the methodology is designed
to detect. In the first budget, the expanded interval covered the whole
$[0,1]$ range at every cycle ($u_c \approx 0.98$, about $21$ times the mean
absolute error), and the correlation between $u_c$ and the actual error was
too weak to distinguish more and less reliable predictions. Accepted at face
value, it would have been read as merely ``conservative''; the Phase~5 checks
instead exposed it as a defect of the input-uncertainty model, a systematic
component propagated as random noise.

With the random/systematic interpretation, the combined uncertainty
($u_c = 0.093$) is of the same order as the actual error
($\bar u_c / \overline{|e|} = 2.0$) and correlates more strongly with it
($r = 0.45$). The intervals are still about twice too wide, because the data
set declares only a total uncertainty without a repeatability or noise
specification, so the random part of the velocity and plain-bearing
acceleration channels can only be bounded from the data. This is a limitation
of the input information, not of the propagation or combination rules. The
point estimate is unaffected by these considerations: $R^2 = 0.948$,
RMSE $= 0.066$, median absolute error $\approx 3.4\%$ of the life range,
confirmed by the blocked split.
```

**Paragrafo rosso su Angrisani** — aggiornare l'ultima frase:

```latex
In the present case, the leverage point is not the number of training
observations but the information on the input uncertainty: a repeatability
or noise specification for the velocity signal and the plain-bearing
accelerometer, or repeated static measurements, would allow the random part
to be evaluated rather than bounded.
```

### 6.7 Configurazione senza pressione

La bozza non la menziona. Se gli autori mantengono la configurazione a 11 canali, basta una riga, per esempio in coda a §5.3:

```latex
Removing the pressure channel halves the systematic part ($0.383$) but leaves
the same-instrument budget almost unchanged ($u_c = 0.089$), since the random
part of that channel is negligible.
```

---

## 7. Supplementary material

**Cosa cambiare.** Aggiungere i documenti successivi alla prima analisi.

```latex
A detailed execution report for the case study documents the environment and
pinned versions, the implementation self-test of the propagation code, the
first budget and its diagnosis, the random/systematic interpretation of the
input uncertainty with its verification, the per-channel uncertainties, the
Monte Carlo convergence analysis, the residual consistency check and the
blocked-split check. A separate report describes the full data set, compares
the declared uncertainties with the manufacturers' specifications and lists
the inconsistencies found with respect to its reference publication. The
package also includes the analysis pipelines, dependency list, per-cycle
results for all held-out cycles, figure-generation script, execution guide
and run logs, enabling exact reproduction of every reported number. The ZeMA
data set is openly available under CC-BY-4.0~\cite{Dorst2021_ZeMA_dataset,ludwig2022_zema_emc_annotated}.
```

---

## 8. Sezione 6 — Conclusioni e open questions

**Validity of the linearity assumption** — "violated in ours" non è più vero. Testo proposto:

```latex
\textit{Validity of the linearity assumption.} The first-order linearity
assumption has been empirically confirmed in some
settings~\cite{Lin2025_GUM_Based_Measurement_Uncertainty_Analysis_Nonlinear_Optical_Angle_Sensor},
including ours once the input uncertainty was correctly interpreted
(Section~\ref{sec:case_study_validation}). The same case shows that a large
LPU--Monte Carlo discrepancy can signal an incorrect input-uncertainty model
rather than an intrinsically nonlinear network, and that it reappears for
large correlated perturbations such as a systematic shift. The assumption has
not been systematically characterized as a function of network size,
activation function type, or input-uncertainty magnitude relative to the
training domain; such an analysis would give practitioners principled a
priori criteria for choosing between analytical and Monte Carlo methods.
```

**Nuova open question** (dopo *Training data uncertainty*):

```latex
\textit{Uncertainty annotation of data sets.} Data sets annotated with
measurement uncertainty are essential to uncertainty-aware machine learning,
but a single declared standard uncertainty per reading does not distinguish
components that vary between observations from those common to all of them,
and the two affect a learned model in different ways. Annotations that
separate random and systematic components, document how each was evaluated,
and state the conditions under which the systematic component can be assumed
stationary would make input budgets evaluable rather than merely boundable.
```

Il paragrafo rosso iniziale delle conclusioni resta valido. Correzioni: "Whereas others work" → "Whereas other works", "guaranties" → "guarantees", punto mancante alla fine del paragrafo su LLM/agent ("…applications").

---

## 9. Correzioni minori indipendenti dal caso di studio

- `\journal{Measurement: Digitalization}`: TODO già segnalato.
- §2: "Must be noted that…" → "It must be noted that…".
- §4 Phase 5: "Reporting consistent with GUM principle" → "principles".
- §5: "(Section~\ref{sec:case_study_validation} illustrates this convergence check for Case Study)" → "for the case study".
- Le note verdi (figure da rifare, simboli) restano azioni degli autori.

---

## 10. Allineamento dei documenti del repository al paper aggiornato

**Stato: eseguito (30 Oct 2026).** Il termine residuo stimato sul validation set ora si chiama `u_calibration` nel testo e nelle tabelle di `case_study_A_report_budget.md`, `case_study_A_report.md`/`.tex` (§11.1 e status update), `README.md` e `case_study_A_guide.md`. Script, log e colonna CSV mantengono il nome `u_model`, indicato con una nota in ogni documento. Le righe che descrivono il budget originale (`u_model² = u_training²`, notazione del paper) restano invariate, con un rimando al budget corretto.

| Documento | Modifica |
|---|---|
| `case_study_A_report_budget.md`, `case_study_A_report.md`/`.tex` §11.1, `README.md` | Il termine residuo da validation `u_model` → `u_calibration` nel testo e nelle tabelle (la colonna CSV `u_model` resta, con una nota). Nel paper `u_model` indica la somma di training, architecture e calibration. |
| `case_study_A_guide.md:103` | Dice ancora che `u_calibration` è fuori scopo e che u_c = √(u_physical² + u_training²); va aggiornata alla combinazione corretta (u_rand, u_training, u_calibration) con rimando al budget report. |
| `case_study_A_report.md` §11.1 | Tabella: "u_model" → "u_calibration"; nessun numero cambia. |
| (eventuale) | Se si sceglie l'opzione (b) di §6.3, aggiungere al budget report la sezione di convergenza MC per il modello `datasheet`. |

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from qet_simulator.data import DEMO_EXAMPLES, build_vocab, encode_text, make_demo_batch
from qet_simulator.model import ClassicalBaseline, QETPipeline
from qet_simulator.quantum import QuantumConfig
from qet_simulator.training import train_demo_models
from qet_simulator.visuals import feature_figure, heatmap_figure, probability_figure, training_curve


st.set_page_config(page_title="QET Simulator", page_icon="⚛️", layout="wide")

st.markdown(
    """
    <style>
      .main-title {font-size:2.05rem;font-weight:750;letter-spacing:-0.03em;}
      .subtle {color: #64748b;font-size:0.92rem;}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18);padding:12px;border-radius:12px;}
      .note {padding:12px 14px;border-left:4px solid #0d9488;background:rgba(13,148,136,.07);border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">⚛️ Quantum-Enhanced Transformer Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Interactive QIA + QSL research pipeline · PennyLane + PyTorch · exact, finite-shot and noisy simulation</div>',
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------- Session defaults -----------------------------
for key, value in {
    "training_result": None,
    "training_meta": None,
    "last_run": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------------------- Sidebar -----------------------------------
st.sidebar.header("Simulator configuration")
st.sidebar.caption("Paper-faithful defaults are available; smaller values make interactive runs faster on CPU.")
n_qubits = st.sidebar.slider("Number of qubits", min_value=2, max_value=6, value=4, step=1)
embedding_dim = st.sidebar.selectbox("Embedding dimension", options=[16, 24, 32, 64], index=0)
max_length = st.sidebar.slider("Maximum tokens", min_value=6, max_value=24, value=12, step=1)
qsl_variant_label = st.sidebar.selectbox(
    "QSL circuit ordering",
    options=["Paper equation (literal order)", "Entangling alternative (CNOT last)"],
    index=0,
    help="The paper-equation version executes the rightmost CNOT chain first, directly on |0...0>. The alternative moves entanglement after local state preparation.",
)
qsl_variant = "paper_equation" if qsl_variant_label.startswith("Paper") else "entangling_variant"
use_positions = st.sidebar.checkbox(
    "Add positional embeddings (extension)",
    value=False,
    help="Off by default because positional embeddings are not present in the manuscript's full-model equations.",
)

st.sidebar.divider()
st.sidebar.subheader("Quantum execution")
mode_label = st.sidebar.radio(
    "Execution backend",
    options=["Ideal statevector", "Finite shots", "Noisy finite shots"],
    index=0,
)
mode = {
    "Ideal statevector": "ideal",
    "Finite shots": "shots",
    "Noisy finite shots": "noisy",
}[mode_label]
shots = st.sidebar.select_slider("Measurement shots", options=[100, 250, 500, 1000, 2000, 5000], value=1000)
depol = st.sidebar.slider("Depolarizing probability per affected wire/gate", 0.0, 0.05, 0.001, 0.001, format="%.3f")
readout = st.sidebar.slider("Readout error probability", 0.0, 0.10, 0.0, 0.005, format="%.3f")
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)

config = QuantumConfig(
    n_qubits=n_qubits,
    mode=mode,
    shots=int(shots),
    depolarizing_probability=float(depol),
    readout_probability=float(readout),
    qsl_variant=qsl_variant,
    seed=int(seed),
)

# -------------------------- Training configuration --------------------------
st.header("1. Train the research demo")
left, right = st.columns([1.35, 1])
with left:
    st.markdown(
        "The included training corpus is a small, balanced sentiment teaching set. "
        "It verifies that gradients pass through the hybrid model; it is **not** AG News, IMDB, SST-2 or SST-5, "
        "and its scores cannot reproduce the paper's reported results."
    )
    train_epochs = st.slider("Training epochs", 1, 6, 2)
    train_batch_size = st.select_slider("Training batch size", options=[2, 4, 5, 10], value=4)
    train_lr = st.select_slider("Learning rate", options=[0.001, 0.003, 0.005, 0.01], value=0.003, format_func=lambda x: f"{x:g}")
    train_button = st.button("Train QET and classical baseline", type="primary", use_container_width=True)
with right:
    st.markdown("**Current architecture settings**")
    st.metric("Qubits", n_qubits)
    st.metric("Embedding dimension", embedding_dim)
    st.metric("Quantum mode for inference", mode_label)
    st.caption("Training uses exact statevector simulation for stable gradients. The selected backend is used during inference and circuit inspection.")

if train_button:
    with st.spinner("Training the QET and classical models on the toy corpus..."):
        try:
            result = train_demo_models(
                quantum_config=config,
                embedding_dim=embedding_dim,
                max_length=max_length,
                epochs=train_epochs,
                batch_size=int(train_batch_size),
                learning_rate=float(train_lr),
                seed=int(seed),
                use_position_embeddings=use_positions,
            )
            st.session_state.training_result = result
            st.session_state.training_meta = {
                "n_qubits": n_qubits,
                "embedding_dim": embedding_dim,
                "max_length": max_length,
                "qsl_variant": qsl_variant,
                "use_positions": use_positions,
                "seed": int(seed),
            }
            st.session_state.last_run = None
            st.success(f"Training finished in {result.elapsed_seconds:.1f} seconds.")
        except Exception as exc:
            st.error("Training failed. See the diagnostic details below.")
            st.exception(exc)

training_result = st.session_state.training_result
training_meta = st.session_state.training_meta

if training_result is not None:
    matches_training_config = training_meta == {
        "n_qubits": n_qubits,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "qsl_variant": qsl_variant,
        "use_positions": use_positions,
        "seed": int(seed),
    }
    if not matches_training_config:
        st.warning(
            "The architecture settings differ from the saved trained model. Restore the settings used during training "
            "or clear the trained model and train again."
        )
    loss_col, metric_col = st.columns([1.5, 1])
    with loss_col:
        st.plotly_chart(training_curve(training_result.qet_history, training_result.classical_history), use_container_width=True)
    with metric_col:
        st.metric("QET training-set accuracy", f"{training_result.qet_accuracy * 100:.1f}%")
        st.metric("Classical training-set accuracy", f"{training_result.classical_accuracy * 100:.1f}%")
        st.caption("Training-set accuracy only; no held-out validation split is used in this tiny demonstration.")
    if st.button("Clear trained models"):
        st.session_state.training_result = None
        st.session_state.training_meta = None
        st.session_state.last_run = None
        st.rerun()

# -------------------------------- Text input ---------------------------------
st.header("2. Run the QET pipeline")
default_sentence = "the movie was excellent and moving"
sentence = st.text_input("Input sentence", value=default_sentence, max_chars=500)
run_button = st.button("Run QET pipeline", type="primary", use_container_width=True)

if run_button:
    if training_result is not None and not matches_training_config:
        st.error("The saved model does not match the current architecture controls. Restore the original settings or clear/retrain it.")
    else:
        texts = [text for text, _label in DEMO_EXAMPLES]
        vocab = training_result.vocab if training_result is not None else build_vocab(texts)
        input_ids, attention_mask, tokens = encode_text(sentence, vocab, max_length)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        qet_model = QETPipeline(
            vocab_size=len(vocab),
            n_classes=2,
            embedding_dim=embedding_dim,
            quantum_config=config,
            dropout=0.0,
            use_position_embeddings=use_positions,
            max_length=max_length,
        )
        classical_model = ClassicalBaseline(
            vocab_size=len(vocab),
            n_classes=2,
            embedding_dim=embedding_dim,
            dropout=0.0,
            use_position_embeddings=use_positions,
            max_length=max_length,
        )
        trained = training_result is not None
        if trained:
            qet_model.load_state_dict(training_result.qet_state)
            classical_model.load_state_dict(training_result.classical_state)
        qet_model.eval()
        classical_model.eval()

        try:
            with st.spinner("Executing token embeddings, QIA, QSL and classification..."):
                start = time.perf_counter()
                with torch.no_grad():
                    qet_logits, trace = qet_model(input_ids, attention_mask, return_trace=True)
                    classical_logits = classical_model(input_ids, attention_mask)
                    qet_probs = torch.softmax(qet_logits, dim=-1)[0].cpu().numpy()
                    classical_probs = torch.softmax(classical_logits, dim=-1)[0].cpu().numpy()
                elapsed = time.perf_counter() - start

            ideal_probs = None
            if mode != "ideal":
                ideal_config = QuantumConfig(
                    n_qubits=n_qubits,
                    mode="ideal",
                    shots=int(shots),
                    depolarizing_probability=float(depol),
                    readout_probability=float(readout),
                    qsl_variant=qsl_variant,
                    seed=int(seed),
                )
                ideal_model = QETPipeline(
                    vocab_size=len(vocab),
                    n_classes=2,
                    embedding_dim=embedding_dim,
                    quantum_config=ideal_config,
                    dropout=0.0,
                    use_position_embeddings=use_positions,
                    max_length=max_length,
                )
                if trained:
                    ideal_model.load_state_dict(training_result.qet_state)
                ideal_model.eval()
                with torch.no_grad():
                    ideal_probs = torch.softmax(ideal_model(input_ids, attention_mask), dim=-1)[0].cpu().numpy()

            st.session_state.last_run = {
                "sentence": sentence,
                "tokens": tokens,
                "trace": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in trace.items()},
                "qet_probs": qet_probs,
                "classical_probs": classical_probs,
                "ideal_probs": ideal_probs,
                "elapsed": elapsed,
                "mode": mode,
                "trained": trained,
                "vocab": vocab,
                "model_config": {
                    "n_qubits": n_qubits,
                    "embedding_dim": embedding_dim,
                    "qsl_variant": qsl_variant,
                    "mode": mode,
                    "shots": int(shots),
                    "depol": float(depol),
                    "readout": float(readout),
                    "seed": int(seed),
                },
                "qet_model": qet_model,
            }
            st.success(f"Pipeline completed in {elapsed:.2f} seconds.")
        except Exception as exc:
            st.error("Pipeline execution failed. Diagnostic details:")
            st.exception(exc)

run = st.session_state.last_run
if run is not None:
    st.subheader("3. Classification output")
    if not run["trained"]:
        st.warning("These predictions use randomly initialized weights. Train the demo models above before interpreting the probabilities.")
    class_names = ["Negative", "Positive"]
    qet_label = class_names[int(np.argmax(run["qet_probs"]))]
    classical_label = class_names[int(np.argmax(run["classical_probs"]))]
    c1, c2, c3 = st.columns(3)
    c1.metric("QET prediction", qet_label, f"{run['qet_probs'].max() * 100:.1f}% confidence")
    c2.metric("Classical prediction", classical_label, f"{run['classical_probs'].max() * 100:.1f}% confidence")
    c3.metric("Execution time", f"{run['elapsed']:.2f} s")

    probability_figure = go.Figure()
    probability_figure.add_trace(go.Bar(x=class_names, y=run["qet_probs"], name=f"QET ({run['mode']})"))
    probability_figure.add_trace(go.Bar(x=class_names, y=run["classical_probs"], name="Classical baseline"))
    if run["ideal_probs"] is not None:
        probability_figure.add_trace(go.Bar(x=class_names, y=run["ideal_probs"], name="QET ideal reference"))
    probability_figure.update_layout(barmode="group", title="Predicted class probabilities", yaxis_title="Probability", yaxis_range=[0, 1], height=370)
    st.plotly_chart(probability_figure, use_container_width=True)

    if run["ideal_probs"] is not None:
        delta = np.abs(run["qet_probs"] - run["ideal_probs"])
        st.info(
            "The selected finite-shot/noisy prediction is compared with an exact ideal reference using the same trained weights. "
            f"Absolute probability differences: negative={delta[0]:.4f}, positive={delta[1]:.4f}."
        )

    st.markdown("**Token sequence**")
    st.write(" · ".join(run["tokens"]))

    trace = run["trace"]
    labels = run["tokens"]
    tab_qia, tab_qsl, tab_state, tab_circuits, tab_data = st.tabs(
        ["QIA analysis", "QSL analysis", "Quantum state", "Circuit diagrams", "Numerical trace"]
    )

    with tab_qia:
        st.markdown("**Quantum-derived token features**")
        st.plotly_chart(
            feature_figure(trace["qia_features"][0, : len(labels)].numpy()[0], "QIA Pauli-Z feature vector — first token", "Pauli-Z"),
            use_container_width=True,
        )
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.plotly_chart(heatmap_figure(trace["kernel"][0, : len(labels), : len(labels)], labels, "Quantum-feature Gram similarity D"), use_container_width=True)
        with q_col2:
            st.plotly_chart(heatmap_figure(trace["interference"][0, : len(labels), : len(labels)], labels, "Phase-modulated norm term I", colorscale="RdBu"), use_container_width=True)
        st.plotly_chart(heatmap_figure(trace["attention"][0, : len(labels), : len(labels)], labels, "Final QIA attention weights A"), use_container_width=True)
        st.latex(r"D=\Phi\Phi^T,\qquad I_{ij}=2\|\phi_i\|\|\phi_j\|\cos(\varphi),\qquad A=\operatorname{softmax}(D+I)")
        st.caption("In this manuscript, I is a phase-scaled feature-norm term. The simulator does not label it as a direct measurement of amplitude interference.")

    with tab_qsl:
        st.markdown("**QSL Pauli-X feature vector — first token**")
        st.plotly_chart(
            feature_figure(trace["qsl_features"][0, : len(labels)].numpy()[0], "QSL Pauli-X expectation values", "Pauli-X"),
            use_container_width=True,
        )
        qsl_matrix = trace["qsl_features"][0, : len(labels)].numpy()
        qsl_fig = go.Figure(go.Heatmap(
            z=qsl_matrix,
            x=[f"q{i}" for i in range(qsl_matrix.shape[1])],
            y=labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar_title="Expectation",
        ))
        qsl_fig.update_layout(title="QSL Pauli-X features across tokens", xaxis_title="Qubit", yaxis_title="Token", height=380)
        st.plotly_chart(qsl_fig, use_container_width=True)
        st.markdown("**Feature projection and residual output**")
        st.write(f"QSL quantum feature shape: `{tuple(trace['qsl_features'].shape)}`")
        st.write(f"Projected token representation shape: `{tuple(trace['qsl_projection'].shape)}`")
        if qsl_variant == "paper_equation":
            st.warning(
                "Paper-equation ordering: the rightmost CNOT chain executes first on |0…0>, so it is inactive. "
                "The following local rotations and Hadamards do not create entanglement. Select the entangling alternative in the sidebar to inspect a distinct corrected circuit."
            )
        else:
            st.success("Entangling alternative selected: local states are prepared before the CNOT chain is applied.")

    with tab_state:
        if run["mode"] == "ideal":
            qet_model = run["qet_model"]
            first_reduced = trace["qia_reduced"][0, 0]
            try:
                state, probs = qet_model.quantum.inspect_qia_state(first_reduced)
                basis_col, amp_col = st.columns(2)
                with basis_col:
                    st.plotly_chart(probability_figure(probs), use_container_width=True)
                with amp_col:
                    amp_fig = go.Figure()
                    basis = [format(i, f"0{n_qubits}b") for i in range(len(state))]
                    amp_fig.add_trace(go.Bar(x=basis, y=np.real(state), name="Real"))
                    amp_fig.add_trace(go.Bar(x=basis, y=np.imag(state), name="Imaginary"))
                    amp_fig.update_layout(title="Statevector amplitudes", barmode="group", xaxis_title="Computational basis", yaxis_title="Amplitude", height=340)
                    st.plotly_chart(amp_fig, use_container_width=True)
                st.caption("Statevector shown for the first token's QIA circuit, with the trained circuit parameters and input encoding.")
            except Exception as exc:
                st.error("Could not construct the ideal statevector view.")
                st.exception(exc)
        else:
            st.info(
                "A finite-shot or noisy mixed-state execution does not have a single exact pure-state vector matching the sampled output. "
                "Switch to Ideal statevector to inspect complex amplitudes. In this mode, the measurement features and probability differences remain available."
            )

    with tab_circuits:
        qet_model = run["qet_model"]
        first_reduced_qia = trace["qia_reduced"][0, 0]
        first_reduced_qsl = trace["qsl_reduced"][0, 0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**QIA quantum feature-map circuit**")
            try:
                st.code(qet_model.quantum.draw_qia_circuit(first_reduced_qia), language="text")
            except Exception as exc:
                st.warning(f"Circuit drawing unavailable: {exc}")
        with c2:
            st.markdown("**QSL circuit**")
            try:
                st.code(qet_model.quantum.draw_qsl_circuit(first_reduced_qsl), language="text")
            except Exception as exc:
                st.warning(f"Circuit drawing unavailable: {exc}")

    with tab_data:
        st.markdown("**Intermediate tensor shapes**")
        shape_rows = []
        for key, value in trace.items():
            if torch.is_tensor(value):
                shape_rows.append({"Stage": key, "Shape": str(tuple(value.shape)), "Dtype": str(value.dtype)})
        st.dataframe(pd.DataFrame(shape_rows), use_container_width=True, hide_index=True)
        st.markdown("**Numerical attention matrix**")
        matrix = trace["attention"][0, : len(labels), : len(labels)].numpy()
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels).round(4), use_container_width=True)

# ----------------------------- Research notes -------------------------------
st.divider()
st.header("Research interpretation and scope")
with st.expander("How the simulator maps to the manuscript", expanded=False):
    st.markdown(
        """
        - **QIA:** the single-head equations are implemented as written: linear reduction, the trainable RX/RZ/RY feature map with two nearest-neighbour CNOT chains, Pauli-Z expectations, the Gram matrix, norm/phase term, row-wise softmax and value reweighting.
        - **QSL:** the manuscript's displayed operator is evaluated using the standard right-to-left operator convention. The alternative CNOT-last option is exposed separately as an experimental circuit correction.
        - **Full pipeline:** embedding → QIA → residual/LayerNorm → QSL projection → residual/LayerNorm → masked mean pooling → classifier.
        - **Training:** the demo trains end-to-end using PyTorch and PennyLane's differentiable exact simulator. Finite-shot/noisy modes are used for inference, not for the demo training loop.
        - **Classical comparison:** single-head scaled dot-product attention plus a position-wise feedforward layer.
        """
    )
with st.expander("Known limitations of this first version", expanded=False):
    st.markdown(
        """
        1. The built-in corpus is a tiny teaching dataset, not a benchmark reproduction.
        2. The simple tokenizer is not BERT tokenization, and no pretrained BERT model is loaded.
        3. The manuscript does not specify positional embeddings. They are off by default; enabling them is an explicit extension.
        4. The noisy backend applies local depolarizing channels after gates and readout error at measurement. It is a generic noise model, not a hardware-calibrated noise profile.
        5. The current simulator implements the single-head QIA path. The manuscript's separate four-head formulation is not included in this first pipeline build.
        6. The paper's QSL subsection includes an internal residual and LayerNorm, while the full-model equation adds a second outer residual/LayerNorm. This simulator follows the full-model equation to avoid silently double-counting residuals.
        7. Results do not establish quantum advantage or quantum speedup. They are simulator outputs for controlled inspection.
        """
    )

st.caption("QET Simulator v0.1.0 · Research prototype · CPU-friendly configuration recommended for local use")

import { requestJson } from "./dashboard/api.js";
import { state } from "./dashboard/state.js";
import { setupUploadArea, uploadFiles, updateRegisterButton } from "./dashboard/uploads.js";
import { activateTab, showTaskInResult, setupResultTabs } from "./dashboard/render.js";
import { fetchDiffusionBaseline, fetchExperimentSummary } from "./dashboard/experiments.js";
import { fetchHistory, fetchRecentLogs, loadHistoryTask } from "./dashboard/history.js";
import { fetchDemoSamples, setupDemoSampleActions } from "./dashboard/demos.js";
import { setupImagePreview } from "./dashboard/imagePreview.js";

async function checkModelStatus() {
    const data = await requestJson("/api/model/status");
    const diffusionReady = Boolean(data.diffusion_available);
    document.getElementById("model-status").textContent = diffusionReady ? "扩散链路已就绪" : "环境待检查";
    document.getElementById("device-status").textContent = data.diffusion_device === "cuda" ? "GPU (CUDA)" : "CPU";
    document.getElementById("method-status").textContent = diffusionReady ? "扩散 + LightGlue" : "运行环境待检查";
    document.getElementById("current-model-status").textContent = `${data.default_model || "ACD_LCM_ADV:model.safetensors"} / 8 steps / 2048 kp / cascade`;
}

async function registerDiffusionImages() {
    try {
        document.getElementById("loading").classList.remove("hidden");
        document.getElementById("results").classList.add("hidden");

        await uploadFiles();
        const registerData = await requestJson("/api/diffusion-register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: state.sessionId,
                steps: Number(document.getElementById("diffusion-steps").value || 8),
                max_keypoints: Number(document.getElementById("lightglue-keypoints").value || 2048),
                extractor_policy: "cascade",
                extractors: ["superpoint", "aliked"],
                match_preprocess: "rgb",
            }),
        });

        showTaskInResult(
            {
                model_name: registerData.results.model_name,
                timings: registerData.timings || {},
                metrics: registerData.results.metrics || {},
                preview_urls: {
                    checkerboard: registerData.checkerboard_url,
                    overlay: registerData.overlay_url,
                    difference: registerData.difference_url,
                    contour: registerData.contour_url,
                    fake_optical: registerData.fake_optical_url,
                    matches: registerData.match_url,
                    sar_transferred_matches: registerData.sar_transfer_match_url,
                    optical_transferred_points: registerData.optical_points_url,
                    registered_preview: registerData.registered_preview_url,
                    sar_condition: registerData.sar_condition_url,
                    real_optical: registerData.real_optical_url,
                    deformation: registerData.deformation_url,
                    download: registerData.registered_url,
                },
            },
            "扩散模型已生成 Fake Optical，并通过级联 LightGlue 完成几何估计。"
        );

        await fetchHistory();
        await fetchRecentLogs();
    } catch (error) {
        document.getElementById("loading").classList.add("hidden");
        alert(`配准流程未完成：${error.message}`);
    }
}

async function logoutUser() {
    await requestJson("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
}

function setupActions() {
    document.getElementById("logout-button").addEventListener("click", logoutUser);
    document.getElementById("diffusion-register-btn").addEventListener("click", registerDiffusionImages);
    document.getElementById("download-button").addEventListener("click", () => {
        if (!state.registeredDownloadUrl) {
            alert("请先完成一次配准或加载一个内置样例。");
            return;
        }
        window.open(state.registeredDownloadUrl, "_blank");
    });

    document.getElementById("history-list").addEventListener("click", (event) => {
        const button = event.target.closest(".history-action");
        if (!button) {
            return;
        }
        if (button.dataset.action === "replay") {
            loadHistoryTask(button.dataset.session);
        } else if (button.dataset.url) {
            window.open(button.dataset.url, "_blank");
        }
    });

    setupUploadArea("sar-upload", "sar-input", "sar-info", (file) => {
        state.sarFile = file;
        updateRegisterButton();
    });

    setupUploadArea("optical-upload", "optical-input", "optical-info", (file) => {
        state.opticalFile = file;
        updateRegisterButton();
    });

    setupResultTabs();
    setupSectionNavigation();
    setupDemoSampleActions();
    setupImagePreview();
}

function setupSectionNavigation() {
    document.querySelectorAll(".dashboard-nav a").forEach((link) => {
        link.addEventListener("click", (event) => {
            const targetHash = link.getAttribute("href");
            if (targetHash === "#match-evidence") {
                event.preventDefault();
                activateTab("tab-matches");
                document.getElementById("result-stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
            } else if (targetHash === "#result-stage") {
                activateTab("tab-overview");
            }
        });
    });
}

setupActions();

checkModelStatus()
    .then(fetchDemoSamples)
    .then(fetchExperimentSummary)
    .then(fetchDiffusionBaseline)
    .then(fetchHistory)
    .then(fetchRecentLogs)
    .catch((error) => {
        console.error("Dashboard init failed:", error);
    });

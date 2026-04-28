import { requestJson } from "./api.js";
import { state } from "./state.js";

export function updateRegisterButton() {
    const disabled = !(state.sarFile && state.opticalFile);
    document.getElementById("diffusion-register-btn").disabled = disabled;
}

export function setupUploadArea(areaId, inputId, infoId, setter) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    const info = document.getElementById(infoId);

    const applyFile = (file) => {
        setter(file);
        info.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        info.classList.remove("hidden");
    };

    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            applyFile(file);
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        area.addEventListener(eventName, (event) => {
            event.preventDefault();
            area.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        area.addEventListener(eventName, (event) => {
            event.preventDefault();
            area.classList.remove("dragover");
        });
    });

    area.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (file) {
            applyFile(file);
        }
    });
}

export async function uploadFiles() {
    const formData = new FormData();
    formData.append("sar", state.sarFile);
    formData.append("optical", state.opticalFile);

    const data = await requestJson("/api/upload", {
        method: "POST",
        body: formData,
    });
    state.sessionId = data.session_id;
    return data;
}

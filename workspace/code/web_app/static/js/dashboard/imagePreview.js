function createPreviewOverlay() {
    const overlay = document.createElement("div");
    overlay.className = "image-preview-overlay hidden";
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML = `
        <div class="image-preview-backdrop" data-preview-close></div>
        <section class="image-preview-dialog" role="dialog" aria-modal="true" aria-label="Image preview">
            <header class="image-preview-header">
                <div>
                    <span class="image-preview-kicker">Image Preview</span>
                    <strong id="image-preview-title">预览图像</strong>
                </div>
                <div class="image-preview-actions">
                    <a id="image-preview-open" class="image-preview-link" href="#" target="_blank" rel="noopener">新窗口打开</a>
                    <button class="image-preview-close" type="button" data-preview-close>关闭</button>
                </div>
            </header>
            <div class="image-preview-stage">
                <img id="image-preview-img" alt="Preview image">
            </div>
        </section>
    `;
    document.body.appendChild(overlay);
    return overlay;
}

function imageTitle(image) {
    const card = image.closest(".image-card");
    const heading = card?.querySelector("h3")?.textContent?.trim();
    return heading || image.alt || "预览图像";
}

function openPreview(image) {
    const overlay = document.getElementById("image-preview-overlay") || createPreviewOverlay();
    overlay.id = "image-preview-overlay";

    const previewImage = overlay.querySelector("#image-preview-img");
    const title = overlay.querySelector("#image-preview-title");
    const openLink = overlay.querySelector("#image-preview-open");
    const src = image.currentSrc || image.src;

    previewImage.src = src;
    previewImage.alt = image.alt || "Preview image";
    title.textContent = imageTitle(image);
    openLink.href = src;

    overlay.classList.remove("hidden");
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("image-preview-open");
}

function closePreview() {
    const overlay = document.getElementById("image-preview-overlay");
    if (!overlay) {
        return;
    }
    overlay.classList.add("hidden");
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("image-preview-open");
}

export function setupImagePreview() {
    document.addEventListener("click", (event) => {
        const closeTarget = event.target.closest("[data-preview-close]");
        if (closeTarget) {
            closePreview();
            return;
        }

        const image = event.target.closest(".image-card img");
        if (!image || !image.getAttribute("src")) {
            return;
        }
        openPreview(image);
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closePreview();
        }
    });
}

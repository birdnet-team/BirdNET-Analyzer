function init() {
    function checkForNewerVersion() {
        let gui_version_element = document.getElementById("current-version")

        if (gui_version_element && gui_version_element.textContent != "main") {
            console.log("Checking for newer version...");

            function sendGetRequest(url) {
                return new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();
                    xhr.open("GET", url);
                    xhr.onload = () => {
                        if (xhr.status === 200) {
                            resolve(xhr.responseText);
                        } else {
                            reject(new Error(`Request failed with status ${xhr.status}`));
                        }
                    };
                    xhr.onerror = () => {
                        reject(new Error("Request failed"));
                    };
                    xhr.send();
                });
            }

            // Not "differs": a build newer than the latest release would be sent back to it.
            function isNewer(latest, current) {
                const latest_parts = latest.split("."), current_parts = current.split(".");

                for (let i = 0; i < Math.max(latest_parts.length, current_parts.length); i++) {
                    const difference = Number(latest_parts[i] || 0) - Number(current_parts[i] || 0);

                    if (difference) {
                        return difference > 0;
                    }
                }

                return false;
            }

            const apiUrl = "https://api.github.com/repos/birdnet-team/BirdNET-Analyzer/releases/latest";

            sendGetRequest(apiUrl)
                .then(response => {
                    const current_version = document.getElementById("current-version").textContent;
                    const response_object = JSON.parse(response);
                    const latest_version = response_object.tag_name.replace(/^v/i, '');

                    if (isNewer(latest_version, current_version)) {
                        const updateNotification = document.getElementById("update-available");

                        updateNotification.style.display = "block";
                        const linkElement = updateNotification.getElementsByTagName("a")[0]
                        linkElement.href = response_object.html_url;
                        linkElement.target = "_blank";
                    }
                })
                .catch(error => {
                    console.error(error);
                });
        }
    }

    function overwriteStyles() {
        console.log("Overwriting styles...");
        const styles = document.createElement("style");
        styles.innerHTML = "@media (width <= 1024px) { .app {max-width: initial !important;}}";
        document.head.appendChild(styles);
    }

    function bindReviewKeyShortcuts() {
        console.log("Binding review key shortcuts...");

        document.addEventListener("keydown", function (event) {
            const posBtn = document.getElementById("positive-button");
            const negBtn = document.getElementById("negative-button");
            const skipBtn = document.getElementById("skip-button");
            const undoBtn = document.getElementById("undo-button");
            const reviewTabBtn = document.getElementById("review-tab-button");

            if (!posBtn || !negBtn) return;

            if (reviewTabBtn.ariaSelected === "false") return;

            if (event.key === "ArrowUp") {
                event.preventDefault();
                posBtn.click();
            } else if (event.key === "ArrowDown") {
                event.preventDefault();
                negBtn.click();
            } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                undoBtn.click();
            } else if (event.key === "ArrowRight") {
                event.preventDefault();
                skipBtn.click();
            }
        });
    }

    function markDetectionsAtPlayhead() {
        const seconds = (text) => {
            const [h, m, sec] = text.trim().split(":");
            return +h * 3600 + +m * 60 + parseFloat(sec);
        };

        let time = null;
        const mark = () => {
            if (time === null) return;

            for (const row of document.querySelectorAll("#single-file-output .virtual-row")) {
                const cells = row.querySelectorAll("[role=gridcell]");
                const start = cells.length > 2 ? seconds(cells[1].textContent) : NaN;
                const end = cells.length > 2 ? seconds(cells[2].textContent) : NaN;
                row.classList.toggle("at-playhead", start <= time && time < end);
            }
        };

        document.addEventListener("signavis-timeupdate", (event) => {
            if (!event.target.closest("#single-file-player")) return;
            time = event.detail.currentTime;
            mark();
        });
        // The grid reuses its rows while scrolling.
        document.addEventListener("scroll", (event) => {
            if (event.target.closest?.("#single-file-output")) requestAnimationFrame(mark);
        }, true);
    }

    checkForNewerVersion();
    markDetectionsAtPlayhead();
    overwriteStyles();
    bindReviewKeyShortcuts();
}
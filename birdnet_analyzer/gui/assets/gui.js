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

            const apiUrl = "https://api.github.com/repos/birdnet-team/BirdNET-Analyzer/releases/latest";

            sendGetRequest(apiUrl)
                .then(response => {
                    const current_version = document.getElementById("current-version").textContent;
                    const response_object = JSON.parse(response);
                    const latest_version = response_object.tag_name.replace(/^v/, '');

                    if (current_version !== latest_version) {
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
        const posBtn = document.getElementById("positive-button");
        const negBtn = document.getElementById("negative-button");
        const skipBtn = document.getElementById("skip-button");
        const undoBtn = document.getElementById("undo-button");

        if (!posBtn || !negBtn) return;

        console.log("Binding review key shortcuts...");

        document.addEventListener("keydown", function (event) {
            const reviewTabBtn = document.getElementById("review-tab-button");

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

    checkForNewerVersion();
    overwriteStyles();
    bindReviewKeyShortcuts();
}
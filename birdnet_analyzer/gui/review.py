import os
import random
from functools import partial

import gradio as gr

import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer import utils

POSITIVE_LABEL_DIR = "Positive"
NEGATIVE_LABEL_DIR = "Negative"
MATPLOTLIB_FIGURE_ID = "review-tab-spectrogram-plot"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def build_review_tab():
    def collect_segments(directory, shuffle=False):
        segments = (
            [
                entry.path
                for entry in os.scandir(directory)
                if (entry.is_file() and not entry.name.startswith(".") and entry.name.rsplit(".", 1)[-1] in cfg.ALLOWED_FILETYPES)
            ]
            if os.path.isdir(directory)
            else []
        )

        if shuffle:
            random.shuffle(segments)

        return segments

    def collect_files(directory):
        return (
            collect_segments(directory),
            collect_segments(os.path.join(directory, POSITIVE_LABEL_DIR)),
            collect_segments(os.path.join(directory, NEGATIVE_LABEL_DIR)),
        )

    def create_log_plot(positives, negatives, fig_num=None):
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.special import expit
        from sklearn import linear_model

        matplotlib.use("agg")

        f = plt.figure(fig_num, figsize=(12, 6))
        f.clear()
        f.tight_layout(pad=0)
        f.set_dpi(300)

        ax = f.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel(f"{loc.localize('review-tab-regression-plot-y-label-false')}/{loc.localize('review-tab-regression-plot-y-label-true')}")
        ax.set_xlabel(loc.localize("review-tab-regression-plot-x-label"))

        x_vals = []
        y_val = []

        for fl in positives + negatives:
            try:
                x_val = float(os.path.basename(fl).split("_", 1)[0])

                if 0 > x_val > 1:
                    continue

                x_vals.append([x_val])
                y_val.append(1 if fl in positives else 0)
            except ValueError:
                pass

        if (len(positives) + len(negatives)) >= 2 and len(set(y_val)) > 1:
            log_model = linear_model.LogisticRegression(C=55)
            log_model.fit(x_vals, y_val)
            Xs = np.linspace(0, 10, 200)
            Ys = expit(Xs * log_model.coef_ + log_model.intercept_).ravel()
            target_ps = [0.85, 0.9, 0.95, 0.99]
            thresholds = [(np.log(target_p / (1 - target_p)) - log_model.intercept_[0]) / log_model.coef_[0][0] for target_p in target_ps]
            p_colors = ["blue", "purple", "orange", "green"]

            for target_p, p_color, threshold in zip(target_ps, p_colors, thresholds, strict=True):
                if threshold <= 1:
                    ax.vlines(
                        threshold,
                        0,
                        target_p,
                        color=p_color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"p={target_p:.2f} threshold>={threshold:.2f}",
                    )
                    ax.hlines(target_p, 0, threshold, color=p_color, linestyle="--", linewidth=0.5)

            ax.plot(Xs, Ys, color="red")
            ax.scatter(thresholds, target_ps, color=p_colors, marker="x")

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            if any(threshold <= 1 for threshold in thresholds):
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if len(y_val) > 0:
            ax.scatter(x_vals, y_val, 2)

        return gr.Plot(value=f, visible=bool(y_val))

    with gr.Tab(loc.localize("review-tab-title"), elem_id="review-tab"):
        review_state = gr.State(
            {
                "input_directory": "",
                "species_list": [],
                "current_species": "",
                "files": [],
                POSITIVE_LABEL_DIR: [],
                NEGATIVE_LABEL_DIR: [],
                "skipped": [],
                "history": [],
            }
        )

        select_directory_btn = gr.Button(loc.localize("review-tab-input-directory-button-label"))

        with gr.Column(visible=False) as review_col:
            with gr.Row():
                species_dropdown = gr.Dropdown(label=loc.localize("review-tab-species-dropdown-label"))
                file_count_matrix = gr.Matrix(
                    headers=[
                        loc.localize("review-tab-file-matrix-todo-header"),
                        loc.localize("review-tab-file-matrix-pos-header"),
                        loc.localize("review-tab-file-matrix-neg-header"),
                    ],
                    interactive=False,
                    elem_id="segments-results-grid",
                )

            with gr.Column() as review_item_col, gr.Row():
                with gr.Column(), gr.Group():
                    spectrogram_image = gr.Plot(label=loc.localize("review-tab-spectrogram-plot-label"), show_label=False)
                    spectrogram_dl_btn = gr.Button("Download spectrogram", size="sm")

                with gr.Column():
                    positive_btn = gr.Button(
                        loc.localize("review-tab-pos-button-label"),
                        elem_id="positive-button",
                        variant="huggingface",
                        icon=os.path.join(SCRIPT_DIR, "assets/arrow_up.svg"),
                    )
                    negative_btn = gr.Button(
                        loc.localize("review-tab-neg-button-label"),
                        elem_id="negative-button",
                        variant="huggingface",
                        icon=os.path.join(SCRIPT_DIR, "assets/arrow_down.svg"),
                    )

                    with gr.Row():
                        undo_btn = gr.Button(
                            loc.localize("review-tab-undo-button-label"),
                            elem_id="undo-button",
                            icon=os.path.join(SCRIPT_DIR, "assets/arrow_left.svg"),
                        )
                        skip_btn = gr.Button(
                            loc.localize("review-tab-skip-button-label"),
                            elem_id="skip-button",
                            icon=os.path.join(SCRIPT_DIR, "assets/arrow_right.svg"),
                        )

                    with gr.Group():
                        review_audio = gr.Audio(type="filepath", sources=[], show_download_button=False, autoplay=True)
                        autoplay_checkbox = gr.Checkbox(True, label=loc.localize("review-tab-autoplay-checkbox-label"))

            no_samles_label = gr.Label(loc.localize("review-tab-no-files-label"), visible=False, show_label=False)
            with gr.Group():
                species_regression_plot = gr.Plot(label=loc.localize("review-tab-regression-plot-label"))
                regression_dl_btn = gr.Button("Download regression", size="sm")

        def update_values(next_review_state, skip_plot=False):
            update_dict = {review_state: next_review_state}

            if not skip_plot:
                update_dict |= {
                    species_regression_plot: create_log_plot(next_review_state[POSITIVE_LABEL_DIR], next_review_state[NEGATIVE_LABEL_DIR], 2),
                }

            if next_review_state["files"]:
                next_file = next_review_state["files"][0]
                update_dict |= {
                    review_audio: gr.Audio(next_file, label=os.path.basename(next_file)),
                    spectrogram_image: utils.spectrogram_from_file(next_file, fig_num=MATPLOTLIB_FIGURE_ID, fig_size=(8, 4)),
                }

            update_dict |= {
                file_count_matrix: [
                    [
                        len(next_review_state["files"]) + len(next_review_state["skipped"]),
                        len(next_review_state[POSITIVE_LABEL_DIR]),
                        len(next_review_state[NEGATIVE_LABEL_DIR]),
                    ],
                ],
                undo_btn: gr.Button(interactive=bool(next_review_state["history"])),
                positive_btn: gr.Button(interactive=bool(next_review_state["files"])),
                negative_btn: gr.Button(interactive=bool(next_review_state["files"])),
                skip_btn: gr.Button(interactive=bool(next_review_state["files"])),
                no_samles_label: gr.Label(visible=not bool(next_review_state["files"])),
                review_item_col: gr.Column(visible=bool(next_review_state["files"])),
                regression_dl_btn: gr.Button(
                    visible=update_dict[species_regression_plot].constructor_args["visible"] if species_regression_plot in update_dict else False
                ),
            }

            return update_dict

        def next_review(next_review_state: dict, target_dir: str | None = None):
            try:
                current_file = next_review_state["files"][0]
            except IndexError as e:
                if next_review_state["input_directory"]:
                    raise gr.Error(loc.localize("review-tab-no-files-error")) from e

                return {review_state: next_review_state}

            if target_dir:
                selected_dir = os.path.join(
                    next_review_state["input_directory"],
                    next_review_state["current_species"] if next_review_state["current_species"] else "",
                    target_dir,
                )

                os.makedirs(selected_dir, exist_ok=True)

                os.rename(
                    current_file,
                    os.path.join(selected_dir, os.path.basename(current_file)),
                )

                next_review_state[target_dir] += [current_file]
                next_review_state["files"].remove(current_file)

                next_review_state["history"].append((current_file, target_dir))
            else:
                next_review_state["skipped"].append(current_file)
                next_review_state["files"].remove(current_file)
                next_review_state["history"].append((current_file, None))

            return update_values(next_review_state)

        def select_subdir(new_value: str, next_review_state: dict):
            if new_value != next_review_state["current_species"]:
                return update_review(next_review_state, selected_species=new_value)

            return {review_state: next_review_state}

        def start_review(next_review_state):
            dir_name = gu.select_folder(state_key="review-input-dir")

            if dir_name:
                next_review_state["input_directory"] = dir_name
                specieslist = [
                    e.name for e in os.scandir(next_review_state["input_directory"]) if e.is_dir() and e.name not in (POSITIVE_LABEL_DIR, NEGATIVE_LABEL_DIR)
                ]

                next_review_state["species_list"] = specieslist

                return update_review(next_review_state)

            return {review_state: next_review_state}

        def try_confidence(filename):
            try:
                val = float(os.path.basename(filename).split("_", 1)[0])

                if 0 > val > 1:
                    return 0

                return val
            except ValueError:
                return 0

        def update_review(next_review_state: dict, selected_species: str | None = None):
            next_review_state["history"] = []
            next_review_state["skipped"] = []

            if selected_species:
                next_review_state["current_species"] = selected_species
            else:
                next_review_state["current_species"] = next_review_state["species_list"][0] if next_review_state["species_list"] else None

            todo_files, positives, negatives = collect_files(
                os.path.join(next_review_state["input_directory"], next_review_state["current_species"])
                if next_review_state["current_species"]
                else next_review_state["input_directory"]
            )

            todo_files = sorted(todo_files, key=try_confidence, reverse=True)

            next_review_state |= {
                "files": todo_files,
                POSITIVE_LABEL_DIR: positives,
                NEGATIVE_LABEL_DIR: negatives,
            }

            update_dict = {
                review_col: gr.Column(visible=True),
                review_state: next_review_state,
                undo_btn: gr.Button(interactive=bool(next_review_state["history"])),
                positive_btn: gr.Button(interactive=bool(next_review_state["files"])),
                negative_btn: gr.Button(interactive=bool(next_review_state["files"])),
                skip_btn: gr.Button(interactive=bool(next_review_state["files"])),
                file_count_matrix: [
                    [
                        len(next_review_state["files"]),
                        len(next_review_state[POSITIVE_LABEL_DIR]),
                        len(next_review_state[NEGATIVE_LABEL_DIR]),
                    ],
                ],
                species_regression_plot: create_log_plot(next_review_state[POSITIVE_LABEL_DIR], next_review_state[NEGATIVE_LABEL_DIR], 2),
            }

            if not selected_species:
                if next_review_state["species_list"]:
                    update_dict |= {
                        species_dropdown: gr.Dropdown(
                            choices=next_review_state["species_list"],
                            value=next_review_state["current_species"],
                            visible=True,
                        )
                    }
                else:
                    update_dict |= {species_dropdown: gr.Dropdown(visible=False)}

            if todo_files:
                update_dict |= {
                    review_item_col: gr.Column(visible=True),
                    review_audio: gr.Audio(value=todo_files[0], label=os.path.basename(todo_files[0])),
                    spectrogram_image: utils.spectrogram_from_file(todo_files[0], fig_num=MATPLOTLIB_FIGURE_ID, fig_size=(8, 4)),
                    no_samles_label: gr.Label(visible=False),
                }
            else:
                update_dict |= {review_item_col: gr.Column(visible=False), no_samles_label: gr.Label(visible=True)}

            update_dict[regression_dl_btn] = gr.Button(visible=update_dict[species_regression_plot].constructor_args["visible"])

            return update_dict

        def undo_review(next_review_state):
            if next_review_state["history"]:
                last_file, last_dir = next_review_state["history"].pop()

                if last_dir:
                    os.rename(
                        os.path.join(
                            next_review_state["input_directory"],
                            next_review_state["current_species"] if next_review_state["current_species"] else "",
                            last_dir,
                            os.path.basename(last_file),
                        ),
                        os.path.join(
                            next_review_state["input_directory"],
                            next_review_state["current_species"] if next_review_state["current_species"] else "",
                            os.path.basename(last_file),
                        ),
                    )

                    next_review_state[last_dir].remove(last_file)
                else:
                    next_review_state["skipped"].remove(last_file)

                was_last_file = not next_review_state["files"]
                next_review_state["files"].insert(0, last_file)

                return update_values(next_review_state, skip_plot=not (was_last_file or last_dir))

            return {
                review_state: next_review_state,
                undo_btn: gr.Button(interactive=bool(next_review_state["history"])),
            }

        def toggle_autoplay(value):
            return gr.Audio(autoplay=value)

        autoplay_checkbox.change(toggle_autoplay, inputs=autoplay_checkbox, outputs=review_audio)

        review_change_output = [
            review_col,
            review_item_col,
            review_audio,
            spectrogram_image,
            species_dropdown,
            no_samles_label,
            review_state,
            file_count_matrix,
            species_regression_plot,
            undo_btn,
            skip_btn,
            positive_btn,
            negative_btn,
            regression_dl_btn,
        ]

        spectrogram_dl_btn.click(partial(gu.download_plot, filename="spectrogram"), show_progress="hidden", inputs=spectrogram_image)
        regression_dl_btn.click(partial(gu.download_plot, filename="regression"), show_progress="hidden", inputs=species_regression_plot)

        species_dropdown.change(
            select_subdir,
            show_progress="full",
            inputs=[species_dropdown, review_state],
            outputs=review_change_output,
        )

        review_btn_output = [
            review_audio,
            spectrogram_image,
            review_state,
            review_item_col,
            no_samles_label,
            file_count_matrix,
            species_regression_plot,
            undo_btn,
            skip_btn,
            positive_btn,
            negative_btn,
            regression_dl_btn,
        ]

        positive_btn.click(
            partial(next_review, target_dir=POSITIVE_LABEL_DIR),
            inputs=review_state,
            outputs=review_btn_output,
            show_progress="full",
            show_progress_on=review_audio,
        )

        negative_btn.click(
            partial(next_review, target_dir=NEGATIVE_LABEL_DIR),
            inputs=review_state,
            outputs=review_btn_output,
            show_progress="full",
            show_progress_on=review_audio,
        )

        skip_btn.click(
            next_review,
            inputs=review_state,
            outputs=review_btn_output,
            show_progress="full",
            show_progress_on=review_audio,
        )

        undo_btn.click(
            undo_review,
            inputs=review_state,
            outputs=review_btn_output,
            show_progress="full",
            show_progress_on=review_audio,
        )

        select_directory_btn.click(
            start_review,
            inputs=review_state,
            outputs=review_change_output,
            show_progress="full",
        )


if __name__ == "__main__":
    gu.open_window(build_review_tab)

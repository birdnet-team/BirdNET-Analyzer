"""Named presets of the settings of a GUI tab.

A preset stores the settings a tab persists between sessions (see
:mod:`birdnet_analyzer.gui.state`) under a name the user chooses, so a configuration
that belongs to a project can be recalled at any time. The multi-file tab can
additionally read its settings back from the ``BirdNET_analysis_params.csv`` a
previous analysis saved next to its results.

Presets of the multi-file tab also remember the species list and custom classifier
files. They are stored as paths: a path that no longer exists when the preset is
applied is skipped with a warning, like every other value the tab cannot take.
"""

import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import gradio as gr

import birdnet_analyzer.gui.localization as loc
from birdnet_analyzer import settings, utils

if TYPE_CHECKING:
    from birdnet_analyzer.gui.state import TabState

# The file entries of a preset. They belong to components that are deliberately not
# persisted between sessions, so they live in the preset only.
SPECIES_FILE_KEY = "species_list_file"
CLASSIFIER_FILE_KEY = "custom_classifier_file"


class PresetControls:
    """The preset controls of a single tab.

    Built in two steps: the constructor renders the controls where it is called,
    :meth:`wire` attaches their handlers once every component of the tab exists.
    """

    def __init__(self, tab: str, with_params_file_loader: bool = False):
        """
        Args:
            tab: The id of the tab the presets belong to, e.g. "multi".
            with_params_file_loader: If True, a button to load the settings from the
                parameters file of a previous analysis is shown.
        """
        self.tab = tab
        self.load_params_file_button = None

        with (
            gr.Accordion(loc.localize("presets-accordion-label"), open=False),
            gr.Group(),
        ):
            with gr.Group(), gr.Row(equal_height=True):
                self.preset_dropdown = gr.Dropdown(
                    choices=settings.list_presets(tab),
                    value=None,
                    label=loc.localize("presets-dropdown-label"),
                    scale=4,
                )
                self.load_button = gr.Button(
                    loc.localize("presets-load-button-label"), variant="primary"
                )
                self.delete_button = gr.Button(
                    loc.localize("presets-delete-button-label")
                )

            with gr.Row(equal_height=True):
                self.name_textbox = gr.Textbox(
                    label=loc.localize("presets-name-textbox-label"),
                    max_lines=1,
                    scale=5.2,  # type: ignore
                )
                self.save_button = gr.Button(loc.localize("presets-save-button-label"))

            if with_params_file_loader:
                self.load_params_file_button = gr.Button(
                    loc.localize("presets-load-params-file-button-label")
                )

    def wire(
        self,
        state: "TabState",
        species_file_input=None,
        classifier_state=None,
        classifier_file_input=None,
        classifier_labels_df=None,
    ):
        """Attaches the handlers to the controls.

        Has to be called after every persisted component of the tab is built. The four
        file components are either all passed (the multi-file tab) or all left out.

        Args:
            state: The persisted settings of the tab.
            species_file_input: The species list file component of the tab, if the
                presets should include the selected species list.
            classifier_state: The state holding the selected custom classifier.
            classifier_file_input: The component showing the selected classifier.
            classifier_labels_df: The list showing the labels of the classifier.
        """
        components = state.components()
        with_files = species_file_input is not None
        file_inputs = [species_file_input, classifier_state] if with_files else []
        file_outputs = (
            [
                species_file_input,
                classifier_state,
                classifier_file_input,
                classifier_labels_df,
            ]
            if with_files
            else []
        )
        outputs = components + file_outputs

        def apply_values(values: dict[str, Any]) -> list:
            values = dict(values)
            species_file = values.pop(SPECIES_FILE_KEY, None) if with_files else None
            classifier_file = (
                values.pop(CLASSIFIER_FILE_KEY, None) if with_files else None
            )

            updates, skipped = state.updates_for(values)

            if with_files:
                updates += _file_updates(species_file, classifier_file, skipped)

            if skipped:
                gr.Warning(
                    loc.localize("presets-skipped-values-warning")
                    + " "
                    + ", ".join(skipped)
                )

            return updates

        def on_save(name, *values):
            name = (name or "").strip()

            if not settings.is_valid_preset_name(name):
                raise gr.Error(loc.localize("presets-invalid-name-error"))

            preset = state.snapshot(values[: len(components)])

            if with_files:
                species_file, classifier_file = values[len(components) :]

                if species_file:
                    preset[SPECIES_FILE_KEY] = str(species_file)
                if classifier_file:
                    preset[CLASSIFIER_FILE_KEY] = str(classifier_file)

            try:
                settings.save_preset(self.tab, name, preset)
            except OSError as e:
                settings.write_error_log(e)
                raise gr.Error(loc.localize("presets-save-failed-error")) from e

            gr.Info(loc.localize("presets-saved-info"))

            return gr.update(choices=settings.list_presets(self.tab), value=name)

        def on_load(name):
            if not name:
                gr.Warning(loc.localize("presets-none-selected-warning"))
                return [gr.skip()] * len(outputs)

            preset = settings.load_preset(self.tab, name)

            if preset is None:
                raise gr.Error(loc.localize("presets-missing-preset-error"))

            updates = apply_values(preset)
            gr.Info(loc.localize("presets-loaded-info"))

            return updates

        def on_delete(name):
            if not name:
                gr.Warning(loc.localize("presets-none-selected-warning"))
                return gr.skip()

            settings.delete_preset(self.tab, name)
            gr.Info(loc.localize("presets-deleted-info"))

            return gr.update(choices=settings.list_presets(self.tab), value=None)

        self.save_button.click(
            on_save,
            inputs=[self.name_textbox, *components, *file_inputs],
            outputs=self.preset_dropdown,
            show_progress="hidden",
        )
        self.load_button.click(
            on_load,
            inputs=self.preset_dropdown,
            outputs=outputs,
            show_progress="hidden",
        )
        self.delete_button.click(
            on_delete,
            inputs=self.preset_dropdown,
            outputs=self.preset_dropdown,
            show_progress="hidden",
        )

        if self.load_params_file_button is not None:

            def on_load_params_file():
                import birdnet_analyzer.gui.utils as gu

                file = gu.select_file(
                    ("CSV (*.csv)",), state_key="analysis-params-file"
                )

                if not file:
                    return [gr.skip()] * len(outputs)

                updates = apply_values(load_analysis_params(file))
                gr.Info(loc.localize("presets-loaded-info"))

                return updates

            self.load_params_file_button.click(
                on_load_params_file, outputs=outputs, show_progress="hidden"
            )


def _file_updates(species_file, classifier_file, skipped: list[str]) -> list:
    """Builds the updates for the file components of the multi-file tab.

    A file that no longer exists is skipped and its component keeps its current value.

    Args:
        species_file: The species list path stored in the preset, if any.
        classifier_file: The classifier path stored in the preset, if any.
        skipped: The keys skipped so far. Extended in place.

    Returns:
        Updates for [species file input, classifier state, classifier file input,
        classifier labels list].
    """
    species_update = gr.skip()
    classifier_updates = [gr.skip(), gr.skip(), gr.skip()]

    if species_file:
        if os.path.isfile(species_file):
            species_update = gr.update(value=species_file)
        else:
            skipped.append(SPECIES_FILE_KEY)

    if classifier_file:
        if os.path.isfile(classifier_file):
            labels = utils.read_classifier_labels(classifier_file)
            classifier_updates = [
                classifier_file,
                gr.update(value=classifier_file, visible=True),
                gr.update(value=labels, visible=True)
                if labels
                else gr.update(visible=False),
            ]
        else:
            skipped.append(CLASSIFIER_FILE_KEY)

    return [species_update, *classifier_updates]


def _species_choice(option: str) -> str:
    # The keys gui.utils localizes the species and model radio labels with.
    return loc.localize(f"species-list-radio-option-{option}")


def load_analysis_params(path: str) -> dict[str, Any]:
    """Reads the settings of a previous analysis back from its parameters file.

    The file is the ``BirdNET_analysis_params.csv`` an analysis saves into its output
    directory. The parameters are translated back into the settings of the multi-file
    tab, undoing the transformations the analysis applied (e.g. the audio speed
    factor back into the slider value). Parameters that cannot be read are left out.

    Args:
        path: The path to the parameters file.

    Returns:
        The settings by key, as `TabState.updates_for` expects them.

    Raises:
        gr.Error: If the file is not an analysis parameters file.
    """
    import csv

    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, UnicodeDecodeError) as e:
        raise gr.Error(loc.localize("presets-invalid-params-file-error")) from e

    if len(rows) < 2:
        raise gr.Error(loc.localize("presets-invalid-params-file-error"))

    params = dict(zip(rows[0], rows[1], strict=False))
    values: dict[str, Any] = {}

    def parse(header: str, key: str, converter):
        raw = params.get(header, "").strip()

        if raw:
            with suppress(ValueError):
                values[key] = converter(raw)

    def to_int(raw):
        return int(float(raw))

    def to_speed_slider(raw):
        speed = float(raw)
        # Undo gui.utils.slider_to_value: factors below 1 sit on the negative side.
        return round(speed) if speed >= 1 else -round(1 / speed)

    parse("Minimum confidence", "confidence_slider", float)
    parse("Sensitivity", "sensitivity_slider", float)
    parse("Segment overlap", "overlap_slider", float)
    parse("Merge consecutive detections", "merge_consecutive_slider", to_int)
    parse("Audio speed", "audio_speed_slider", to_speed_slider)
    parse("Bandpass filter minimum", "fmin_number", to_int)
    parse("Bandpass filter maximum", "fmax_number", to_int)
    parse("Species filter threshold", "sf_thresh_number", float)
    parse("Batch size", "batch_size_number", to_int)
    parse("Number of producers", "producers_number", to_int)
    parse("Number of workers", "workers_number", to_int)
    parse("Top N", "top_n_input", to_int)
    parse("Latitude", "lat_number", float)
    parse("Longitude", "lon_number", float)
    parse("Week", "week_number", to_int)
    parse("Locale", "locale_dropdown", str)

    if "Top N" in params:
        values["use_top_n_checkbox"] = "top_n_input" in values

        # With top N in use the analysis ran without a confidence threshold and
        # stored 0, which is not a value the confidence slider offers.
        if values["use_top_n_checkbox"]:
            values.pop("confidence_slider", None)

    if "Week" in params:
        values["yearlong_checkbox"] = "week_number" not in values

    for header, key in (
        ("Result type(s)", "output_type_checkboxgroup"),
        ("Additional columns", "additional_columns_checkboxgroup"),
    ):
        if header in params:
            values[key] = [
                entry.strip() for entry in params[header].split(",") if entry.strip()
            ]

    if params.get("Split tables", "").strip() in ("True", "False"):
        values["split_tables_checkbox"] = params["Split tables"].strip() == "True"

    species_file = params.get("Species list file", "").strip()
    classifier_file = params.get("Custom classifier path", "").strip()

    if species_file:
        values["species_list_radio"] = _species_choice("custom-list")
        values[SPECIES_FILE_KEY] = species_file
    elif "lat_number" in values and "lon_number" in values:
        values["species_list_radio"] = _species_choice("predict-list")
    elif "Species list file" in params:
        values["species_list_radio"] = _species_choice("all")

    if classifier_file:
        values["model_selection_radio"] = _species_choice("custom-classifier")
        values[CLASSIFIER_FILE_KEY] = classifier_file
    elif params.get("Model", "").strip() == "perch":
        values["model_selection_radio"] = _species_choice("use-perch")
    elif params.get("Model", "").strip():
        # Not localized, must match gui.utils._USE_BIRDNET_2_4.
        values["model_selection_radio"] = "BirdNET 2.4"

    # A file without a reasonable number of readable parameters is not an analysis
    # parameters file.
    if len(values) < 3:
        raise gr.Error(loc.localize("presets-invalid-params-file-error"))

    return values

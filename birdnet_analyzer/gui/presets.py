"""Named presets of the settings of a GUI tab.

A preset stores the settings a tab persists between sessions (see
:mod:`birdnet_analyzer.gui.state`) under a name the user chooses, so a configuration
that belongs to a project can be recalled at any time. Both tabs can additionally
read their settings back from the parameters file a previous run saved: the analysis
from the ``birdnet.analyze-params.csv`` in its output directory, the training from
the ``<classifier>.birdnet.train-params.csv`` next to the trained classifier. The
loaders also understand the files written before the 2.x releases
(``BirdNET_analysis_params.csv`` and ``<classifier>_Params.csv``), which held the
same parameters in one column per parameter instead of one row.

Presets of the multi-file tab also remember the species list and custom classifier
files. They are stored as paths: a path that no longer exists when the preset is
applied is skipped with a warning, like every other value the tab cannot take.
"""

import os
from typing import TYPE_CHECKING, Any

import gradio as gr

import birdnet_analyzer.gui.localization as loc
from birdnet_analyzer import params, settings, utils

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

    def __init__(self, tab: str, params_loader=None, params_button_label=None):
        """
        Args:
            tab: The id of the tab the presets belong to, e.g. "multi".
            params_loader: Reads the parameters file of a previous run into settings,
                e.g. :func:`load_analysis_params`. If given, a button to load such a
                file is shown, labelled with ``params_button_label``.
            params_button_label: The label of the load-from-file button.
        """
        self.tab = tab
        self.load_params_file_button = None
        self._params_loader = params_loader

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

            if params_loader is not None:
                self.load_params_file_button = gr.Button(params_button_label)

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
                    ("CSV (*.csv)",), state_key=f"{self.tab}-params-file"
                )

                if not file:
                    return [gr.skip()] * len(outputs)

                updates = apply_values(self._params_loader(file))
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


def _load_params_file(loader, path: str) -> dict[str, Any]:
    try:
        return loader(path)
    except ValueError as e:
        raise gr.Error(loc.localize("presets-invalid-params-file-error")) from e


def _rename(kwargs: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Renames run parameters to the keys of the components showing them."""
    return {
        component_key: kwargs[key]
        for key, component_key in mapping.items()
        if key in kwargs
    }


def _speed_to_slider(speed: float) -> int:
    # Undo gui.utils.slider_to_value: factors below 1 sit on the negative side.
    return round(speed) if speed >= 1 else -round(1 / speed)


def load_analysis_params(path: str) -> dict[str, Any]:
    """Reads the settings of a previous analysis back from its parameters file.

    Maps the parameters read by :func:`birdnet_analyzer.params.load_analysis_params`
    onto the settings of the multi-file tab, undoing the transformations the GUI
    applies when it starts an analysis (e.g. the audio speed factor back into the
    slider value).

    Args:
        path: The path to the parameters file.

    Returns:
        The settings by key, as `TabState.updates_for` expects them.

    Raises:
        gr.Error: If the file is not an analysis parameters file.
    """
    kwargs = _load_params_file(params.load_analysis_params, path)
    values = _rename(
        kwargs,
        {
            "sensitivity": "sensitivity_slider",
            "overlap": "overlap_slider",
            "merge_consecutive": "merge_consecutive_slider",
            "fmin": "fmin_number",
            "fmax": "fmax_number",
            "sf_thresh": "sf_thresh_number",
            "batch_size": "batch_size_number",
            "n_producers": "producers_number",
            "n_workers": "workers_number",
            "top_n": "top_n_input",
            "lat": "lat_number",
            "lon": "lon_number",
            "week": "week_number",
            "locale": "locale_dropdown",
            "split_tables": "split_tables_checkbox",
            "rtype": "output_type_checkboxgroup",
            "additional_columns": "additional_columns_checkboxgroup",
        },
    )

    if "audio_speed" in kwargs:
        values["audio_speed_slider"] = _speed_to_slider(kwargs["audio_speed"])

    values["use_top_n_checkbox"] = "top_n" in kwargs

    # With top N in use the analysis ran without a confidence threshold and stored 0,
    # which is not a value the confidence slider offers.
    if "top_n" not in kwargs and "min_conf" in kwargs:
        values["confidence_slider"] = kwargs["min_conf"]

    values["yearlong_checkbox"] = "week" not in kwargs

    if "slist" in kwargs:
        values["species_list_radio"] = _species_choice("custom-list")
        values[SPECIES_FILE_KEY] = kwargs["slist"]
    elif "lat" in kwargs and "lon" in kwargs:
        values["species_list_radio"] = _species_choice("predict-list")
    else:
        values["species_list_radio"] = _species_choice("all")

    if "classifier" in kwargs:
        values["model_selection_radio"] = _species_choice("custom-classifier")
        values[CLASSIFIER_FILE_KEY] = kwargs["classifier"]
    elif kwargs.get("model") == "perch":
        values["model_selection_radio"] = _species_choice("use-perch")
    elif "model" in kwargs:
        # Not localized, must match gui.utils._USE_BIRDNET_2_4.
        values["model_selection_radio"] = "BirdNET 2.4"

    return values


def load_train_params(path: str) -> dict[str, Any]:
    """Reads the settings of a previous training run back from its parameters file.

    Maps the parameters read by :func:`birdnet_analyzer.params.load_train_params`
    onto the settings of the train tab.

    Args:
        path: The path to the parameters file.

    Returns:
        The settings by key, as `TabState.updates_for` expects them.

    Raises:
        gr.Error: If the file is not a training parameters file.
    """
    kwargs = _load_params_file(params.load_train_params, path)
    values = _rename(
        kwargs,
        {
            "classifier_name": "classifier_name_textbox",
            "model_formats": "output_format_checkboxgroup",
            "model_save_mode": "model_save_mode_radio",
            "fmin": "fmin_number",
            "fmax": "fmax_number",
            "crop_mode": "crop_mode_radio",
            "overlap": "crop_overlap_slider",
            "autotune": "autotune_checkbox",
            "autotune_trials": "autotune_trials_number",
            "autotune_n_splits": "autotune_folds_number",
            "autotune_n_repeats": "autotune_repeats_number",
            "epochs": "epochs_number",
            "batch_size": "batch_size_number",
            "learning_rate": "learning_rate_number",
            "hidden_units": "hidden_units_number",
            "dropout": "dropout_number",
            "label_smoothing": "use_label_smoothing_checkbox",
            "mixup": "use_mixup_checkbox",
            "use_focal_loss": "use_focal_loss_checkbox",
            "focal_loss_gamma": "focal_loss_gamma_slider",
            "focal_loss_alpha": "focal_loss_alpha_slider",
            "upsampling_mode": "upsampling_mode_radio",
            "upsampling_ratio": "upsampling_ratio_slider",
        },
    )

    if "audio_speed" in kwargs:
        values["audio_speed_slider"] = _speed_to_slider(kwargs["audio_speed"])

    return values

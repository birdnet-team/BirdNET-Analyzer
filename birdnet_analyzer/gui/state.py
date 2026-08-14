"""Persistence of the setting values a user last worked with in the GUI.

Every tab keeps its own values under its own key in ``state.json``, so changing e.g.
the confidence in the multi-file tab leaves the one in the single-file tab untouched.
A value is written back as soon as the user edits it and is restored when the
component is built, so the GUI comes up showing the values of the last session.

Restored values are validated against the component they belong to. A value that does
not fit -- because the settings file was written by an older version, in a different
language, on a platform offering different models, or was edited by hand -- is
discarded in favour of the default the component is built with. That way a stale
settings file can never keep a tab from starting up.
"""

from typing import Any, NamedTuple

import gradio as gr

from birdnet_analyzer import settings

# Every persisted component with the default it was built with, so the settings tab can
# offer a reset. Populated while the tabs are built, in build order.
_PERSISTED: list[tuple[gr.components.Component, Any]] = []


class _Field(NamedTuple):
    """A persisted component and the constraints its values are validated against."""

    component: gr.components.Component
    default: Any
    choices: Any
    minimum: float | None
    maximum: float | None


def _validate(
    value,
    default,
    choices=None,
    minimum: float | None = None,
    maximum: float | None = None,
):
    """Checks a persisted value against the component it is restored into.

    Args:
        value: The persisted value.
        default: The value the component would be built with.
        choices: The choices of the component, if it has any. Either plain values or
            (label, value) pairs, as accepted by gradio.
        minimum: The lowest value the component accepts, if it has a lower bound.
        maximum: The highest value the component accepts, if it has an upper bound.

    Returns:
        The persisted value if the component accepts it, otherwise the default.
    """
    if choices is not None:
        allowed = {
            choice[1] if isinstance(choice, list | tuple) else choice
            for choice in choices
        }
        # A CheckboxGroup holds a list of choices, every other component a single one.
        # Which of the two it is shows in the default.
        if isinstance(default, list):
            fits = isinstance(value, list) and all(v in allowed for v in value)
        else:
            fits = not isinstance(value, list) and value in allowed

        return value if fits else default

    if isinstance(default, bool):
        return value if isinstance(value, bool) else default

    if isinstance(default, int | float):
        # bool is an int, but a checked checkbox is not a number.
        if isinstance(value, bool) or not isinstance(value, int | float):
            return default

        in_range = (minimum is None or value >= minimum) and (
            maximum is None or value <= maximum
        )

        return value if in_range else default

    if isinstance(default, str):
        return value if isinstance(value, str) else default

    # Without a default to compare against there is nothing to validate the value with.
    return default


class TabState:
    """The persisted settings of a single GUI tab.

    The values are read once, when the tab is built, and are written back one by one as
    the user edits them.
    """

    def __init__(self, tab: str) -> None:
        """
        Args:
            tab (str): The id of the tab the settings belong to, e.g. "multi".
        """
        self.tab = tab
        self._values = settings.get_tab_settings(tab)
        self._fields: dict[str, _Field] = {}

    def get(
        self,
        key: str,
        default,
        choices=None,
        minimum: float | None = None,
        maximum: float | None = None,
    ):
        """Reads a persisted value without building a component for it.

        Args:
            key (str): The name of the setting inside the tab.
            default: The value to fall back to if none was persisted or the persisted
                one is not valid.
            choices: The choices the value has to be one of, if there are any.
            minimum: The lowest accepted value, if there is a lower bound.
            maximum: The highest accepted value, if there is an upper bound.

        Returns:
            The persisted value, or the default.
        """
        if key not in self._values:
            return default

        return _validate(
            self._values[key],
            default,
            choices=choices,
            minimum=minimum,
            maximum=maximum,
        )

    def persist(self, key: str, constructor, **kwargs):
        """Builds a gradio component that remembers its value across sessions.

        The component is built with the value the user last set, falling back to the
        ``value`` it is given here, and saves its value whenever the user changes it.

        Args:
            key (str): The name of the setting inside the tab.
            constructor: The gradio component class, e.g. ``gr.Slider``.
            **kwargs: The arguments to build the component with. ``value`` is the
                default the component falls back to.

        Returns:
            The created component.
        """
        default = kwargs.get("value")
        kwargs["value"] = self.get(
            key,
            default,
            choices=kwargs.get("choices"),
            minimum=kwargs.get("minimum"),
            maximum=kwargs.get("maximum"),
        )

        component = constructor(**kwargs)
        _PERSISTED.append((component, default))
        self._fields[key] = _Field(
            component,
            default,
            kwargs.get("choices"),
            kwargs.get("minimum"),
            kwargs.get("maximum"),
        )

        # Only user edits are persisted. Values a component receives from another
        # component's event handler are derived from settings that are persisted
        # themselves, or from data the user has to select again anyway (the embeddings
        # tab e.g. overrides the settings of an existing database).
        trigger = (
            component.release if isinstance(component, gr.Slider) else component.input
        )
        trigger(
            lambda value, key=key: settings.set_tab_setting(self.tab, key, value),
            inputs=component,
            show_progress="hidden",
            queue=False,
        )

        return component

    def components(self) -> list[gr.components.Component]:
        """Returns every component persisted in this tab, in build order."""
        return [field.component for field in self._fields.values()]

    def snapshot(self, values) -> dict[str, Any]:
        """Pairs the current component values with their setting keys.

        Args:
            values: The current values of `components()`, in the same order.

        Returns:
            The values by setting key, e.g. to be saved as a preset.
        """
        return dict(zip(self._fields, values, strict=True))

    def updates_for(self, values: dict) -> tuple[list[dict], list[str]]:
        """Builds the component updates that show a set of persisted values.

        Every value is validated against the component it belongs to, exactly like a
        value restored from a previous session. A value that does not fit -- or a key
        no component was built for -- is skipped and the component keeps its current
        value. The applied values are persisted, so they survive a restart just like
        user edits.

        Args:
            values: The values to show, by setting key. E.g. a saved preset.

        Returns:
            A tuple of the updates (one per component, in build order) and the keys
            whose values could not be applied.
        """
        updates = []
        applied = {}
        skipped = [key for key in values if key not in self._fields]

        for key, field in self._fields.items():
            if key not in values:
                updates.append(gr.update())
                continue

            value = values[key]
            accepted = _validate(
                value,
                field.default,
                choices=field.choices,
                minimum=field.minimum,
                maximum=field.maximum,
            )

            # _validate hands the value object itself back iff the component takes it.
            if accepted is value:
                updates.append(gr.update(value=value))
                applied[key] = value
            else:
                updates.append(gr.update())
                skipped.append(key)

        settings.update_tab_settings(self.tab, applied)

        return updates, skipped


def persisted_components() -> list[gr.components.Component]:
    """Returns every component built through a `TabState`, in build order."""
    return [component for component, _ in _PERSISTED]


def reset_to_defaults() -> list[dict]:
    """Discards the persisted settings of all tabs and restores the default values.

    The components are reset in place, so the change handlers of the tabs pick the
    reset up and show the components belonging to a default value again.

    Returns:
        An update per component built through a `TabState`, in build order.
    """
    settings.reset_tab_settings()

    return [gr.update(value=default) for _, default in _PERSISTED]

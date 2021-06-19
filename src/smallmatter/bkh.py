from typing import List, Union

import bokeh.io
import bokeh.plotting
import bokeh.resources
from bokeh.models import ColumnDataSource, HoverTool


class BokehPlotter(object):
    blue = "#539ac2"
    red = "#dc4a4e"

    def __init__(self, df, title: str, x_label, y_label, hover_tooltips, **kwargs):
        """Generate Bokeh plot with hover tooltips.

        :param df: input dataframe
        :param title: plot title
        :param x_label: for bokeh plot.xaxis.axis_label
        :param y_label: for bokeh plot.yaxis.axis_label
        :param hover_tooltips: Bokeh's hover tooltips
        :param kwargs: for bokeh.plotting.figure
        """
        self.df = df

        # Create a Bokeh plot
        self.src = ColumnDataSource(df)
        self.title = title
        # Default figure setting
        plot_kwargs = dict(
            plot_width=640,
            plot_height=480,
            title=title,
            tools="pan,wheel_zoom,box_zoom,crosshair,reset,hover,save",
            min_border=1,
        )
        # Override with user setting
        plot_kwargs.update(kwargs)
        self.plot = bokeh.plotting.figure(**plot_kwargs)

        # Customize x_label & y_label
        self.plot.xaxis.axis_label = x_label
        self.plot.yaxis.axis_label = y_label

        # Settings for hover
        self.hover_tooltips = hover_tooltips
        self.hovered_glyphs: Union[List, str] = "all"

        # More space on top & below, to prevent cutting-off tooltips
        self.plot.min_border_top = 64
        self.plot.min_border_bottom = 64

    def plot_lines(self):
        """Draw two lines: a diagonal line, and the actual roc line."""
        g = self.plot.line(x="x", y="y", source=self.src, alpha=0.8, line_width=3, color=self.blue)

        # Override parent's setting: only enable hover tooltip on roc line.
        self.hovered_glyphs = [g]

    def add_hover(self):
        """Add hover tooltips for all glyph except for those in exclusion list."""
        hover = self.plot.select({"type": HoverTool})
        hover.mode = "vline"
        hover.tooltips = self.hover_tooltips
        if self.hovered_glyphs != "all":
            hover.renderers = self.hovered_glyphs

    def gen_plot(self):
        """Draw plot and save to html."""
        self.plot_lines()
        self.add_hover()

    def save_html(self, ofname):
        """Save the current plot to a stand-alone html file."""
        bokeh.io.save(self.plot, filename=ofname, resources=bokeh.resources.INLINE, title=self.title)

from Tkinter import BOTH, NW, S, N, E, W  # Directions
from Tkinter import Canvas, Frame  # Tk Widgets
from Tkinter import HORIZONTAL  # Orientations
from Tkinter import Tk
from ttk import Scrollbar

from PIL import Image, ImageTk


class Example(Frame):
	def __init__(self, parent):
		Frame.__init__(self, parent)
		self.parent = parent
		self._pos()
		self._init_ui()
	def _pos(self):
		"""
			Position window  of size 500x500at the centre of the screen.
		"""
		sw = self.parent.winfo_screenwidth()
		sh = self.parent.winfo_screenheight()
		w = 500
		h = 500
		x = (sw - w) / 2
		y = (sh - h) / 2
		self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))
	def _init_ui(self):
		# Load an image
		self.img = ImageTk.PhotoImage(Image.open(r"images\dna5.png"))
		# Define a canvas in a frame
		frame = Frame(self)
		c = Canvas(frame, bg="white", height=475, width=475)
		# Display the image in the canvas
		c.create_image(0, 0, image=self.img, anchor=NW)
		# Y-scrollbar
		yscrollbar = Scrollbar(frame, command=c.yview)
		c.configure(yscrollcommand=yscrollbar.set)
		# X-scrollbar
		xscrollbar = Scrollbar(frame, orient=HORIZONTAL, command=c.xview)
		c.configure(xscrollcommand=xscrollbar.set)
		# Display widgets using grid layout.
		frame.grid(row=0, column=0)
		yscrollbar.grid(row=0, column=2, sticky=S + N)
		xscrollbar.grid(row=2, column=0, sticky=W + E)
		c.grid(row=0, column=0)
		self.pack(fill=BOTH, expand=1)
def main():
	root = Tk()
	Example(root)
	root.mainloop()
if __name__ == '__main__':
	main()
	

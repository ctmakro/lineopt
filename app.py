import wx
import json, pickle
from collections import namedtuple
import cv2
import numpy as np

class AppFrame(wx.Frame):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ui_init()
        self.project_init()
        self.ui_update()

    # setup UI
    def ui_init(self,):
        # panel = wx.Panel(self)
        #
        # st = wx.StaticText(panel, label="Helloworld", pos=(25,25))
        #
        # font = st.GetFont()
        # font.PointSize = 12
        # font = font.Bold()
        # st.SetFont(font)

        # parent of everything
        panel = wx.Panel(self)

        # shorthand
        # self.panel = panel

        # root of all sizer(positioner)
        root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(root_sizer)
        self.root_sizer = root_sizer

        # left side
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        leftup_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel, 'Project')
        leftmid_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel, 'Image Adj')
        leftdn_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel, 'Algorithm')

        # right side
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        im = wx.Image(256,256)
        self.image_display = wx.StaticBitmap(panel, -1, wx.Bitmap(im))
        right_sizer.Add(self.image_display)

        root_sizer.Add(left_sizer, 0, wx.EXPAND|wx.ALL, 10)
        root_sizer.Add(right_sizer, 0, wx.EXPAND|wx.ALL&(~wx.LEFT), 10)

        left_sizer.Add(leftup_sizer,0,wx.TOP|wx.EXPAND, 0)
        left_sizer.Add(leftmid_sizer,0,wx.TOP|wx.EXPAND, 5)
        left_sizer.Add(leftdn_sizer,0,wx.TOP|wx.EXPAND, 5)

        def add_button(text, callback=None, sizer=leftdn_sizer):
            button = wx.Button(panel, label=text)
            sizer.Add(button, 0, wx.ALL, 3)
            if callback is not None: self.Bind(wx.EVT_BUTTON, callback, button)
            return button

        def add_label(text, sizer=leftdn_sizer):
            st = wx.StaticText(panel, label=text)
            sizer.Add(st, 0, wx.ALL, 3)
            return st

        # # numeric updn ctrl
        # spin = wx.SpinCtrl(panel, -1, min=0, max=5, initial=1)
        #
        # add_label('Brightness',sizer=leftmid_sizer)
        # leftmid_sizer.Add(spin, 0,wx.ALL,3)


        add_button('Open Project', self.MenuOpenProject, leftup_sizer)
        add_button('Save Project', self.MenuSaveProject, leftup_sizer)
        add_button('Load Image', self.ButtonLoadImage, leftup_sizer)
        add_button('Reload Image', self.ButtonReloadImage, leftup_sizer)

        add_button('say hi', lambda e:wx.MessageBox('wassup yo'))
        add_button('yo', lambda e:wx.MessageBox('bitch'))
        add_label('this thing rocks')
        add_button('dummy')
        add_label('')
        add_button('separated')

        self.init_menu_bar()

        self.CreateStatusBar()

    # update UI to reflect current status
    def ui_update(self):
        self.SetTitle(
            'Painter {} ({})'.format(
                self.project_state['version'],
                self.project_state['source'],
        ))

        self.root_sizer.Fit(self)

    # default(blank) state of project
    # this dictionary contains the full state of current project.
    def project_init(self,):
        self.project_state = {
            'version':'0.1',
            'source':None,
        }

        # if this dictionary contains only text, floats, dicts and lists, it can be serialized using JSON(human readable). if it has to contain other datatypes, use pickle(not human-readable) instead.

    # what to display on the right
    def set_displayed_image(self, img):
        # expect numpy array as input

        # BGR->RGB
        img = np.flip(img,2)

        h,w = img.shape[0:2]

        # get rid of strides
        img = img.flatten()

        # create bitmap object
        bmp = wx.Bitmap.FromBuffer(w, h, img.data)

        self.image_display.SetBitmap(bmp)

    # setup menu
    def init_menu_bar(self):
        menuitem = namedtuple('MenuItem',['parent','text','shortcut','tip','callback'])

        menuitems = []
        def ai(*a):
            menuitems.append(menuitem(*a))

        # all menu items and the parameters to create them
        ai('File','Open Project','Ctrl-O',None,self.MenuOpenProject)
        ai('File','Save Project','Ctrl-S',None,self.MenuSaveProject)
        ai('File','Save Project As','Ctrl-Shift-S',None,self.MenuSaveProjectAs)
        ai('File','Exit','Ctrl-W',None,self.MenuExit)

        # now create them one by one.
        menus = {}
        menuBar = wx.MenuBar()

        for i in menuitems:
            if i.parent not in menus:
                newmenu = wx.Menu()
                menus[i.parent] = newmenu
                menuBar.Append(newmenu, '&'+i.parent)
            menu = menus[i.parent]
            tip = i.tip or i.text
            fulltext = '&'+i.text+'\t'+ (i.shortcut or '')
            newitem = menu.Append(-1, fulltext, tip)
            self.Bind(wx.EVT_MENU, i.callback, newitem)

        self.SetMenuBar(menuBar)

    # project to file
    def write_project_to_file(self, filename):
        try:
            with open(filename, 'wb') as f:
                # json.dump(self.project_state, f)
                pickle.dump(self.project_state, f)
        except Exception as e:
            wx.MessageBox(repr(e))
            return False
        else:
            self.project_state['source'] = filename
            self.ui_update()
            return True

    # project from file
    def read_project_from_file(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.project_state = pickle.load(f)
                # self.project_state = json.load(f)
        except Exception as e:
            wx.MessageBox(repr(e))
            return False
        else:
            self.project_state['source'] = filename
            self.read_image_from_setting()
            self.ui_update()
            return True

    # file dialogs
    def MenuOpenProject(self, event):
        fd = wx.FileDialog(
            parent=self,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            message='Open Project',
            wildcard = 'Pickled Python Objects (*.pickle)|*.pickle',
        )

        if fd.ShowModal() == wx.ID_OK:
            filename = fd.GetPath()
            self.read_project_from_file(filename)

    def MenuSaveProjectAs(self, event):
        fd = wx.FileDialog(
            parent=self,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            message='Save Project',
            wildcard = 'Pickled Python Objects (*.pickle)|*.pickle',
        )

        if fd.ShowModal() == wx.ID_OK:
            filename = fd.GetPath()
            self.write_project_to_file(filename)

    def MenuSaveProject(self, event):
        filename = self.project_state['source']
        if filename is None:
            self.MenuSaveProjectAs(event)
        else:
            self.write_project_to_file(filename)

    def MenuExit(self, event):
        self.Close()

    def ButtonLoadImage(self, event):
        fd = wx.FileDialog(
            parent=self,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            message='Open Project',
            wildcard = 'Images (*.jpg;*.png)|*.jpg;*.png',
        )

        if fd.ShowModal() == wx.ID_OK:
            filename = fd.GetPath()
            self.project_state['image_path'] = filename
            self.read_image_from_setting()

    def ButtonReloadImage(self, event):
        if self.read_image_from_setting() == False:
            self.ButtonLoadImage(event)

    def read_image_from_file(self,path):
        img = cv2.imread(path)
        if img is not None:
            self.target_image = img
            self.set_displayed_image(self.target_image)
            self.ui_update()
        else:
            wx.MessageBox('Failed to read image.')

    def read_image_from_setting(self):
        if 'image_path' in self.project_state:
            self.read_image_from_file(self.project_state['image_path'])
            return True
        else:
            return False

# a parameter that can be tuned.
class Parameter(namedtuple('Parameter',['name','default','min','max','step'])):
    def get(self):
        return self.default

# a plugin that holds a bunch of parameters.
class Parametrized:
    def __init__(self):
        self.parameters = []
        self.paradict = {}

    def add_param(self, *a):
        param = Parameter(*a)
        self.parameters.append(param)
        self.paradict[param.name] = param

    def __getattr__(self,name):
        if name in self.paradict:
            return self.paradict[name]
        else:
            return super().__getattr__(name)

class BrightnessContrast(Parametrized):
    def __init__(self):
        super().__init__()
        self.add_param('Brightness',0,-10,10,0.5)
        self.add_param('Contrast',0,-10,10,0.5)

    def filter(self, img, params):
        img = img.astype('int32')
        img += int(self.Brightness.get()*10)
        img = np.clip(img, 0, 255).astype('uint8')
        return img

i = BrightnessContrast()
print(i.Brightness.get())

if __name__ == '__main__':

    app = wx.App()
    frm = AppFrame(None, title='')
    frm.Show()
    app.MainLoop()

import wx
import json, pickle
from collections import namedtuple

class AppFrame(wx.Frame):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ui_init()
        self.project_init()
        self.ui_update()

    # setup UI
    def ui_init(self,):
        # pnl = wx.Panel(self)
        #
        # st = wx.StaticText(pnl, label="Helloworld", pos=(25,25))
        #
        # font = st.GetFont()
        # font.PointSize = 12
        # font = font.Bold()
        # st.SetFont(font)

        self.init_menu_bar()

        self.CreateStatusBar()
        self.SetStatusText('some status text here')

    # update UI to reflect current status
    def ui_update(self):
        self.SetTitle(
            'Painter {} ({})'.format(
                self.project_state['version'],
                self.project_state['source'],
        ))

    # default(blank) state of project
    # this dictionary contains the full state of current project.
    def project_init(self,):
        self.project_state = {
            'version':'0.1',
            'source':None,
        }

        # if this dictionary contains only text, floats, dicts and lists, it can be serialized using JSON(human readable). if it has to contain other datatypes, use pickle(not human-readable) instead.

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
            self.ui_update()
            return True

    # file dialogs
    def MenuOpenProject(self, event):
        fd = wx.FileDialog(
            parent=self,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            message='Open Project',
            wildcard = 'Pickled Python Objects (.pickle)|.pickle',
        )

        if fd.ShowModal() == wx.ID_OK:
            filename = fd.GetPath()
            self.read_project_from_file(filename)

    def MenuSaveProjectAs(self, event):
        fd = wx.FileDialog(
            parent=self,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            message='Save Project',
            wildcard = 'Pickled Python Objects (.pickle)|.pickle',
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

if __name__ == '__main__':

    app = wx.App()
    frm = AppFrame(None, title='')
    frm.Show()
    app.MainLoop()

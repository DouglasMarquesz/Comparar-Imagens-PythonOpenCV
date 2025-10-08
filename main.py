#Arquivo com a aba inicial
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from interface import Ui_MainWindow  # importando o arquivo convertido

class JanelaPrincipal(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # monta a interface
        self.setWindowTitle("Exemplo PySide6")

        # exemplo: conectar botão
        self.pushButton.clicked.connect(self.clicou)

    def clicou(self):
        print("Botão clicado!")

app = QApplication(sys.argv)
janela = JanelaPrincipal()
janela.show()
app.exec()
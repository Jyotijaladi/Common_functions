import java.awt.*;
import java.awt.event.*;
import java.applet.*;

public class MouseEvents extends Applet implements MouseListener, MouseMotionListener {
    String msg = "";

    public void init() {
        addMouseListener(this);
        addMouseMotionListener(this);
    }

    public void paint(Graphics g) {
        g.drawString(msg, 202, 20);
    }

    @Override
    public void mousePressed(MouseEvent me) {
        msg = "Mouse pressed";
        repaint();
    }

    @Override
    public void mouseClicked(MouseEvent me) {
        msg = "Mouse clicked";
        repaint();
    }

    @Override
    public void mouseExited(MouseEvent me) {
        msg = "Mouse exited";
        repaint();
    }

    @Override
    public void mouseMoved(MouseEvent me) {
        msg = "Mouse moved";
        repaint();
    }

    @Override
    public void mouseDragged(MouseEvent me) {
        msg = "Mouse dragged";
        repaint();
    }

    @Override
    public void mouseEntered(MouseEvent me) {
        // Empty method implementation
    }
    @Override
    public void mouseReleased(MouseEvent me) {
        // Empty method implementation
    }
}

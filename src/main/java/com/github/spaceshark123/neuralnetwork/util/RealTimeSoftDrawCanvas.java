package com.github.spaceshark123.neuralnetwork.util;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;

public class RealTimeSoftDrawCanvas extends JPanel {
    private BufferedImage canvas;
    private Graphics2D g2d;
    private int brushSize = 2;
    private final int canvasSize = 28;
    //private final int scale = 10; // Scale factor for pixelated display
    private double[][] pixelValues;
    private int lastX, lastY;
    private boolean isDrawing;
    private Timer timer;

    public RealTimeSoftDrawCanvas() {
        this.canvas = new BufferedImage(canvasSize, canvasSize, BufferedImage.TYPE_BYTE_GRAY);
        this.g2d = canvas.createGraphics();
        this.g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        this.g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        this.g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f / brushSize));
        clearCanvas();

        setPreferredSize(new Dimension(canvasSize * 10, canvasSize * 10));
        setBackground(Color.BLACK);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                isDrawing = true;
                lastX = e.getX() * canvasSize / getWidth();
                lastY = e.getY() * canvasSize / getHeight();
                draw(lastX, lastY);
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                isDrawing = false;
            }
        });

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isDrawing) {
                    int currentX = e.getX() * canvasSize / getWidth();
                    int currentY = e.getY() * canvasSize / getHeight();
                    drawLine(lastX, lastY, currentX, currentY);
                    lastX = currentX;
                    lastY = currentY;
                }
            }
        });

        // Initialize pixel values
        pixelValues = new double[canvasSize][canvasSize];

        // Timer to update pixel values every 100 milliseconds
        timer = new Timer(100, e -> updatePixelValues());
        timer.start();
    }

    public void stopTimer() {
        timer.stop();
    }

    private void draw(int x, int y) {
        g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f / brushSize)); // Ensure correct composite for drawing
        g2d.setColor(Color.WHITE);
        g2d.fillOval(x - brushSize / 2, y - brushSize / 2, brushSize, brushSize);
        repaint();
    }

    private void drawLine(int x1, int y1, int x2, int y2) {
        g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f / brushSize)); // Ensure correct composite for drawing
        g2d.setColor(Color.WHITE);
        int dx = Math.abs(x2 - x1);
        int dy = Math.abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx - dy;
        while (true) {
            g2d.fillOval(x1 - brushSize / 2, y1 - brushSize / 2, brushSize, brushSize);
            if (x1 == x2 && y1 == y2) {
                break;
            }
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        int newWidth = getWidth();
        int newHeight = getHeight();
        int scaleX = newWidth / canvasSize;
        int scaleY = newHeight / canvasSize;
        int scale = Math.min(scaleX, scaleY);

        int offsetX = (newWidth - canvasSize * scale) / 2;
        int offsetY = (newHeight - canvasSize * scale) / 2;

        g2.setColor(getBackground());
        g2.fillRect(0, 0, newWidth, newHeight);

        g2.drawImage(canvas, offsetX, offsetY, canvasSize * scale, canvasSize * scale, null);
    }

    public void clearCanvas() {
        g2d.setComposite(AlphaComposite.Src); // Reset composite to opaque
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f / brushSize)); // Restore drawing composite
        repaint();
    }

    private void updatePixelValues() {
        BufferedImage scaledImage = new BufferedImage(canvasSize, canvasSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = scaledImage.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(canvas, 0, 0, canvasSize, canvasSize, null);
        g2.dispose();

        for (int y = 0; y < canvasSize; y++) {
            for (int x = 0; x < canvasSize; x++) {
                int rgb = scaledImage.getRGB(x, y) & 0xFF;
                pixelValues[y][x] = (rgb / 255.0);
            }
        }
    }

    public double[][] getPixelArray() {
        return pixelValues;
    }
}
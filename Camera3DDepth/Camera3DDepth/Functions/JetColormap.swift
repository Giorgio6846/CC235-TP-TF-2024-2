//
//  JetColormap.swift
//  Camera3DDepth
//
//  Created by Giorgio Mancusi on 10/4/24.
//

import Foundation
import UIKit


func applyJetColormap(to grayscaleImage: UIImage) -> UIImage? {
    guard let cgImage = grayscaleImage.cgImage else { return nil }
    
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let bitsPerComponent = 8
    
    var pixelData: [UInt8] = Array(repeating: 0, count: width * height * bytesPerPixel)
    
    guard let context = CGContext(data: &pixelData, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
        return nil
    }
    
    context.draw(cgImage, in: CGRect(x: 0, y:0, width: width, height:height))
    
    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = (y * width + x) * bytesPerPixel
            let grayValue = Float(pixelData[pixelIndex]) / 255.0
            
            let jetColor = jetColorForValue(grayValue)
            
            pixelData[pixelIndex] = UInt8(jetColor.red * 255)
            pixelData[pixelIndex + 1] = UInt8(jetColor.green * 255)
            pixelData[pixelIndex + 2] = UInt8(jetColor.blue * 255)
        
        }
    }

    guard let outputCGImage = context.makeImage() else { return nil }
    return UIImage(cgImage: outputCGImage)

}

func jetColorForValue(_ value: Float) -> (red: Float, green: Float, blue: Float) {
    let fourValue = 4 * value
    
    let r = min(fourValue - 1.5, -fourValue + 4.5)
    let g = min(fourValue - 0.5, -fourValue + 3.5)
    let b = min(fourValue + 0.5, -fourValue + 2.5)
    
    return (max(0, min(r,1)), max(0, min(g,1)), max(0, min(b,1)))
}

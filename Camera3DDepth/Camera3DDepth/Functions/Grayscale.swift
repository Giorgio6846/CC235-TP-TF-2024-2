//
//  File.swift
//  Camera3DDepth
//
//  Created by Giorgio Mancusi on 10/4/24.
//

import Foundation
import UIKit
import ARKit

func createGrayscaleImage(from depthData: [vector_float3]) -> UIImage? {
    guard let minDepth = depthData.min(by: {$0.z < $1.z})?.z,
          let maxDepth = depthData.max(by: {$0.z < $1.z})?.z else {
        return nil
    }
    
    let normalizedDepth = depthData.map { ($0.z - minDepth) / (maxDepth - minDepth)}
    
    let width = Int(sqrt(Double(depthData.count)))
    let height = width
    let colorSapace = CGColorSpaceCreateDeviceGray()
    let bitmapInfo = CGImageAlphaInfo.none.rawValue
    let bytesPerPixel = 1
    let bytesPerRow = width * bytesPerPixel
    var pixelData = normalizedDepth.map { UInt8($0*255)}

    //See if it works
    guard let providerRef = CGDataProvider(data: NSData(bytes: &pixelData, length: pixelData.count) as Data as CFData) else {
        return nil
    }
    
    guard let cgImage = CGImage(width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bitsPerPixel: 8 * bytesPerPixel,
                                bytesPerRow: bytesPerRow,
                                space: colorSapace,
                                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                                provider: providerRef,
                                decode: nil,
                                shouldInterpolate: false,
                                intent: .defaultIntent) else {
        return nil
    }
    
    return UIImage(cgImage: cgImage)
    
}

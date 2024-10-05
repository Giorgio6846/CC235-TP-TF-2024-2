//
//  ImageDepthServer.swift
//  Camera3DDepth
//
//  Created by Giorgio Mancusi on 10/4/24.
//

import Foundation
import ARKit
import UIKit

func sendDepthImagetoServer(depthData: [vector_float3], capturedImage: CVPixelBuffer, ip: String) {
    
    let jsonArray = depthData.map {["x": $0.x, "y": $0.y, "z": $0.z]}
    guard let depthJsonData = try? JSONSerialization.data(withJSONObject: jsonArray, options: []) else {
        print("Error al convertir los datos de profundidad a JSON")
        return
    }

    let ciImage = CIImage(cvPixelBuffer: capturedImage)
    let context = CIContext()
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
        print("Error al convertir CVPixelBuffer a CGImage")
        return
    }
    
    let uiImage = UIImage(cgImage: cgImage)
    guard let imageData = uiImage.jpegData(compressionQuality: 1) else {
        print("Error al convertir UIImage a JPEG")
        return
    }
    
    guard let url = URL(string: "http://\(ip)/face-depth-data") else {
        print("URL invalida")
        return
    }
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = "Boundary-\(UUID().uuidString)"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"depthData\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: application/json\r\n\r\n".data(using: .utf8)!)
    body.append(depthJsonData)
    body.append("\r\n".data(using: .utf8)!)

    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n".data(using: .utf8)!)

    body.append("--\(boundary)--\r\n".data(using: .utf8)!)

    request.httpBody = body

    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error al enviar los datos: \(error.localizedDescription)")
            return
        }

        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
            print("Datos enviados con exito")
        } else {
            print("Error al enviar los datos. Respuesta: \(String(describing: response))")
        }
        
    }
    
    task.resume()

}

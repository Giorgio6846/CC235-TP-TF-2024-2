//
//  CameraView.swift
//  Camera3DDepth
//
//  Created by Giorgio Mancusi on 10/4/24.
//

import SwiftUI
import ARKit
import UIKit
import SceneKit

struct ARViewContainer: UIViewRepresentable {
    
    @Binding var showJetColorPreview: Bool
    @Binding var jetImage: UIImage?

    let arView = ARSCNView()
    
    func makeUIView(context: Context) -> ARSCNView {
        arView.delegate = context.coordinator
        
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        arView.session.run(configuration)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.showJetColorPreview = showJetColorPreview
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, ARSCNViewDelegate {
        var parent: ARViewContainer
        var showJetColorPreview:Bool = false
        var ipAddress: String = "10.0.1.10:5050"
        
        init(_ parent: ARViewContainer) {
            self.parent = parent
        }
        
        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let faceAnchor = anchor as? ARFaceAnchor,
                  let currentFrame = parent.arView.session.currentFrame else { return }
            
            let depthData = faceAnchor.geometry.vertices
            let capturedImage = currentFrame.capturedImage
            
            
            
            sendDepthImagetoServer(depthData: depthData, capturedImage: capturedImage, ip: ipAddress)
            
            if showJetColorPreview {
                if let grayscaleImage = createGrayscaleImage(from: depthData) {
                    parent.jetImage = applyJetColormap(to: grayscaleImage)
                }
            }
            
        }
    }

}
